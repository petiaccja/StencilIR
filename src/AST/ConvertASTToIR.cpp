#include "ConvertASTToIR.hpp"

#include "ASTNodes.hpp"
#include "ASTTypes.hpp"
#include "IRGenerator.hpp"
#include "SymbolTable.hpp"

#include <Dialect/Stencil/IR/StencilOps.hpp>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>

#include <algorithm>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <ranges>
#include <stack>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <typeindex>
#include <unordered_map>

using namespace std::string_literals;


//------------------------------------------------------------------------------
// Utilities
//------------------------------------------------------------------------------

mlir::Location ConvertLocation(mlir::OpBuilder& builder, const std::optional<ast::Location>& location) {
    if (location) {
        auto fileattr = builder.getStringAttr(location->file);
        return mlir::FileLineColLoc::get(fileattr, location->line, location->col);
    }
    return builder.getUnknownLoc();
}

struct TypeConversionOptions {
    enum {
        TENSOR,
        MEMREF,
    } bufferType = MEMREF;
};

mlir::Type ConvertType(mlir::OpBuilder& builder, types::Type type, const TypeConversionOptions& options = {}) {
    struct {
        mlir::OpBuilder& builder;
        const TypeConversionOptions& options;
        mlir::Type operator()(const types::FundamentalType& type) const {
            switch (type.type) {
                case types::FundamentalType::SINT8: return builder.getIntegerType(8, true);
                case types::FundamentalType::SINT16: return builder.getIntegerType(16, true);
                case types::FundamentalType::SINT32: return builder.getIntegerType(32, true);
                case types::FundamentalType::SINT64: return builder.getIntegerType(64, true);
                case types::FundamentalType::UINT8: return builder.getIntegerType(8, false);
                case types::FundamentalType::UINT16: return builder.getIntegerType(16, false);
                case types::FundamentalType::UINT32: return builder.getIntegerType(32, false);
                case types::FundamentalType::UINT64: return builder.getIntegerType(64, false);
                case types::FundamentalType::SSIZE: return builder.getIndexType();
                case types::FundamentalType::USIZE: return builder.getIndexType();
                case types::FundamentalType::FLOAT32: return builder.getF32Type();
                case types::FundamentalType::FLOAT64: return builder.getF64Type();
                case types::FundamentalType::BOOL: return builder.getI1Type();
            }
            throw std::invalid_argument("Unknown type.");
        }
        mlir::Type operator()(const types::FieldType& type) const {
            const mlir::Type elementType = (*this)(type.elementType);

            constexpr auto offset = mlir::ShapedType::kDynamicStrideOrOffset;
            std::vector<int64_t> shape(type.numDimensions, mlir::ShapedType::kDynamicSize);
            std::vector<int64_t> strides(type.numDimensions, mlir::ShapedType::kDynamicStrideOrOffset);
            auto strideMap = mlir::makeStridedLinearLayoutMap(strides, offset, builder.getContext());

            if (options.bufferType == TypeConversionOptions::MEMREF) {
                return mlir::MemRefType::get(shape, elementType, strideMap);
            }
            else {
                return mlir::RankedTensorType::get(shape, elementType);
            }
        }
    } visitor{ builder, options };
    return std::visit(visitor, type);
}



//------------------------------------------------------------------------------
// Generator
//------------------------------------------------------------------------------


class StencilIRGenerator : public IRGenerator<ast::Node, mlir::Operation*, std::vector<mlir::Value>, StencilIRGenerator,
                                              ast::Print,
                                              ast::Add,
                                              ast::Sub,
                                              ast::Mul,
                                              ast::Module,
                                              ast::Stencil,
                                              ast::Return,
                                              ast::Apply,
                                              ast::SymbolRef,
                                              ast::Index,
                                              ast::Jump,
                                              ast::Sample,
                                              ast::JumpIndirect,
                                              ast::SampleIndirect,
                                              ast::Constant<float>,
                                              ast::Constant<double>,
                                              ast::Constant<int8_t>,
                                              ast::Constant<int16_t>,
                                              ast::Constant<int32_t>,
                                              ast::Constant<int64_t>,
                                              ast::Constant<uint8_t>,
                                              ast::Constant<uint16_t>,
                                              ast::Constant<uint32_t>,
                                              ast::Constant<uint64_t>,
                                              ast::AllocTensor,
                                              ast::Dim> {
    using IROperation = mlir::Operation*;
    using IRValue = std::vector<mlir::Value>;
    using Generated = std::tuple<IROperation, IRValue>;

    mutable mlir::OpBuilder builder;
    mutable SymbolTable<std::string, mlir::Value> symbolTable;

public:
    StencilIRGenerator(mlir::MLIRContext& context) : builder(&context) {}

private:
    using IRGenerator::Generate;

public:
    template <class Func>
    void InsertInBlock(mlir::Block& block, Func func) const {
        auto previousInsertionPoint = builder.saveInsertionPoint();
        builder.setInsertionPointToEnd(&block);
        func();
        builder.restoreInsertionPoint(previousInsertionPoint);
    }

    static mlir::Value GetSingleResult(const Generated& generated) {
        const auto& [op, values] = generated;
        if (values.size() != 1) {
            throw std::invalid_argument("Operation did not produce a single result.");
        }
        return values.front();
    }

    static Generated FromOp(mlir::Operation* op) {
        auto results = op->getResults();
        return { op, { results.begin(), results.end() } };
    }

    //--------------------------------------------------------------------------
    // Arithmetic and misc
    //--------------------------------------------------------------------------

    template <class T>
    auto Generate(const ast::Constant<T>& node) const -> Generated {
        if constexpr (std::is_floating_point_v<T>) {
            const auto type = std::is_same_v<T, float> ? builder.getF32Type() : builder.getF64Type();
            auto op = builder.create<mlir::arith::ConstantFloatOp>(ConvertLocation(builder, node.location),
                                                                   mlir::APFloat(node.value),
                                                                   type);
            return FromOp(op);
        }
        else if constexpr (std::is_integral_v<T>) {
            constexpr int numBits = sizeof(T) * 8;
            const auto type = node.type ? ConvertType(builder, node.type.value()) : mlir::Type(builder.getIntegerType(numBits));
            using Unsigned = std::make_unsigned_t<decltype(node.value)>;
            const uint64_t unsignedValue = std::bit_cast<Unsigned>(node.value);
            auto op = builder.create<mlir::arith::ConstantOp>(ConvertLocation(builder, node.location),
                                                              builder.getIntegerAttr(type, std::bit_cast<int64_t>(unsignedValue)),
                                                              type);
            return FromOp(op);
        }
        throw std::invalid_argument("Cannot lower this constant type.");
    }

    auto Generate(const ast::Print& node) const -> Generated {
        const auto argument = GetSingleResult(Generate(*node.argument));

        auto op = builder.create<stencil::PrintOp>(ConvertLocation(builder, node.location),
                                                   argument);
        return FromOp(op);
    }

    auto Generate(const ast::Add& node) const -> Generated {
        auto lhs = GetSingleResult(Generate(*node.lhs));
        auto rhs = GetSingleResult(Generate(*node.rhs));
        auto op = builder.create<mlir::arith::AddFOp>(ConvertLocation(builder, node.location),
                                                      lhs,
                                                      rhs);
        return FromOp(op);
    }

    auto Generate(const ast::Sub& node) const -> Generated {
        auto lhs = GetSingleResult(Generate(*node.lhs));
        auto rhs = GetSingleResult(Generate(*node.rhs));
        auto op = builder.create<mlir::arith::SubFOp>(ConvertLocation(builder, node.location),
                                                      lhs,
                                                      rhs);
        return FromOp(op);
    }

    auto Generate(const ast::Mul& node) const -> Generated {
        auto lhs = GetSingleResult(Generate(*node.lhs));
        auto rhs = GetSingleResult(Generate(*node.rhs));
        auto op = builder.create<mlir::arith::MulFOp>(ConvertLocation(builder, node.location),
                                                      lhs,
                                                      rhs);
        return FromOp(op);
    }

    //--------------------------------------------------------------------------
    // Structure
    //--------------------------------------------------------------------------

    auto Generate(const ast::Stencil& node) const -> Generated {
        const TypeConversionOptions typeOptions = {
            .bufferType = TypeConversionOptions::TENSOR
        };

        // Create operation
        std::vector<mlir::Type> inputTypes{};
        for (auto& param : node.parameters) {
            inputTypes.push_back(ConvertType(builder, param.type, typeOptions));
        }
        std::vector<mlir::Type> resultTypes{};
        for (auto& result : node.results) {
            resultTypes.push_back(ConvertType(builder, result, typeOptions));
        }
        const auto functionType = builder.getFunctionType(mlir::TypeRange{ inputTypes }, mlir::TypeRange{ resultTypes });
        const auto loc = ConvertLocation(builder, node.location);
        auto stencilOp = builder.create<stencil::StencilOp>(loc,
                                                            node.name,
                                                            functionType,
                                                            mlir::APInt(64, node.numDimensions));

        // Create function body

        symbolTable.RunInScope([&, this] {
            auto& kernelFuncBlock = *stencilOp.addEntryBlock();

            for (size_t i = 0; i < node.parameters.size(); ++i) {
                symbolTable.Assign(node.parameters[i].name, kernelFuncBlock.getArgument(i));
            }

            InsertInBlock(kernelFuncBlock, [&] {
                for (auto& statement : node.body) {
                    Generate(*statement);
                }
            });
        },
                               std::any(node));

        return FromOp(stencilOp);
    }

    auto Generate(const ast::Return& node) const -> Generated {
        const auto loc = ConvertLocation(builder, node.location);
        std::vector<mlir::Value> values;
        for (const auto& value : node.values) {
            values.push_back(GetSingleResult(Generate(*value)));
        }
        auto op = builder.create<stencil::ReturnOp>(loc, values);
        return FromOp(op);
    }

    auto Generate(const ast::Apply& node) const -> Generated {
        const auto callee = mlir::StringRef(node.callee);

        std::vector<mlir::Value> outputs;
        for (auto& target : node.outputs) {
            outputs.push_back(GetSingleResult(Generate(*target)));
        }

        std::vector<mlir::Value> inputs;
        for (auto& argument : node.inputs) {
            inputs.push_back(GetSingleResult(Generate(*argument)));
        }

        std::vector<mlir::Value> offsets;
        for (auto& offset : node.offsets) {
            offsets.push_back(GetSingleResult(Generate(*offset)));
        }

        auto op = builder.create<stencil::ApplyOp>(ConvertLocation(builder, node.location),
                                                   callee,
                                                   inputs,
                                                   outputs,
                                                   offsets,
                                                   node.static_offsets);

        return FromOp(op);
    }

    auto Generate(const ast::Module& node) const -> Generated {
        auto moduleOp = builder.create<mlir::ModuleOp>(ConvertLocation(builder, node.location));

        auto loc = ConvertLocation(builder, node.location);
        builder.setInsertionPointToEnd(moduleOp.getBody());

        const TypeConversionOptions typeOptions = {
            .bufferType = TypeConversionOptions::TENSOR
        };

        // Render kernels
        for (auto& kernel : node.kernels) {
            Generate(*kernel);
        }

        // Create a main function.
        std::vector<mlir::Type> inputTypes{};
        for (auto& param : node.parameters) {
            inputTypes.push_back(ConvertType(builder, param.type, typeOptions));
        }
        const auto mainFuncType = builder.getFunctionType(mlir::TypeRange{ inputTypes }, mlir::TypeRange{});
        auto mainFuncOp = builder.create<mlir::func::FuncOp>(loc, "main", mainFuncType);

        // Create body.
        mlir::Block& mainBlock = *mainFuncOp.addEntryBlock();
        symbolTable.RunInScope([&, this] {
            for (size_t i = 0; i < node.parameters.size(); ++i) {
                auto blockArg = mainBlock.getArgument(i);
                if (blockArg.getType().isa<mlir::MemRefType>()) {
                    InsertInBlock(mainBlock, [&] {
                        auto tensorValue = builder.create<mlir::bufferization::ToTensorOp>(loc, blockArg);
                        symbolTable.Assign(node.parameters[i].name, tensorValue);
                    });
                }
                else {
                    symbolTable.Assign(node.parameters[i].name, blockArg);
                }
            }

            InsertInBlock(mainBlock, [&] {
                for (auto& statement : node.body) {
                    Generate(*statement);
                }
                builder.create<mlir::func::ReturnOp>(loc);
            });
        });

        return FromOp(moduleOp);
    }

    auto Generate(const ast::SymbolRef& node) const -> Generated {
        const auto value = symbolTable.Lookup(node.name);
        if (!value) {
            throw std::invalid_argument("Undefined symbol: " + node.name);
        }
        return { nullptr, { *value } };
    }

    //--------------------------------------------------------------------------
    // Kernel intrinsics
    //--------------------------------------------------------------------------

    auto Generate(const ast::Index& node) const -> Generated {
        const auto loc = ConvertLocation(builder, node.location);

        const auto currentFunc = std::any_cast<ast::Stencil>(symbolTable.Info());
        const int64_t numDims = currentFunc.numDimensions;
        const auto indexType = mlir::VectorType::get({ numDims }, builder.getIndexType());

        auto op = builder.create<stencil::IndexOp>(loc, indexType);
        return FromOp(op);
    }

    auto Generate(const ast::Jump& node) const -> Generated {
        const auto loc = ConvertLocation(builder, node.location);
        const auto index = GetSingleResult(Generate(*node.index));
        const auto offset = builder.getI64ArrayAttr(node.offset);
        auto op = builder.create<stencil::JumpOp>(loc, index.getType(), index, offset);
        return FromOp(op);
    }

    auto Generate(const ast::Sample& node) const -> Generated {
        const auto loc = ConvertLocation(builder, node.location);
        const auto field = GetSingleResult(Generate(*node.field));
        const auto index = GetSingleResult(Generate(*node.index));

        auto fieldType = field.getType();
        if (!fieldType.isa<mlir::ShapedType>()) {
            throw std::invalid_argument("SampleOp must be used to sample fields.");
        }
        auto elementType = fieldType.dyn_cast<mlir::ShapedType>().getElementType();

        auto op = builder.create<stencil::SampleOp>(loc, elementType, field, index);
        return FromOp(op);
    }

    auto Generate(const ast::JumpIndirect& node) const -> Generated {
        const auto loc = ConvertLocation(builder, node.location);

        const auto index = GetSingleResult(Generate(*node.index));
        const auto dimension = builder.getIndexAttr(node.dimension);
        const auto map = GetSingleResult(Generate(*node.map));
        const auto mapElement = GetSingleResult(Generate(*node.mapElement));

        auto op = builder.create<stencil::JumpIndirectOp>(loc, index.getType(), index, dimension, map, mapElement);
        return FromOp(op);
    }

    auto Generate(const ast::SampleIndirect& node) const -> Generated {
        const auto loc = ConvertLocation(builder, node.location);
        auto index = GetSingleResult(Generate(*node.index));
        auto dimension = builder.getIndexAttr(node.dimension);
        auto field = GetSingleResult(Generate(*node.field));
        auto fieldElement = GetSingleResult(Generate(*node.fieldElement));

        auto fieldType = field.getType();
        if (!fieldType.isa<mlir::ShapedType>()) {
            throw std::invalid_argument("SampleIndirectOp must be used to sample fields.");
        }
        auto elementType = fieldType.dyn_cast<mlir::ShapedType>().getElementType();

        auto op = builder.create<stencil::SampleIndirectOp>(loc, elementType, index, dimension, field, fieldElement);
        return FromOp(op);
    }

    //--------------------------------------------------------------------------
    // Tensor
    //--------------------------------------------------------------------------

    auto Generate(const ast::Dim& node) const -> Generated {
        auto loc = ConvertLocation(builder, node.location);
        const mlir::Value tensor = GetSingleResult(Generate(*node.field));
        const mlir::Value index = GetSingleResult(Generate(*node.index));
        if (tensor.getType().isa<mlir::MemRefType>()) {
            return FromOp(builder.create<mlir::memref::DimOp>(loc, tensor, index));
        }
        else {
            return FromOp(builder.create<mlir::tensor::DimOp>(loc, tensor, index));
        }
    }

    auto Generate(const ast::AllocTensor& node) const -> Generated {
        auto loc = ConvertLocation(builder, node.location);

        const auto elementType = ConvertType(builder, node.elementType);
        std::vector<mlir::Value> sizes;
        std::vector<int64_t> shape;
        for (const auto& size : node.sizes) {
            sizes.push_back(GetSingleResult(Generate(*size)));
            shape.push_back(mlir::ShapedType::kDynamicSize);
        }
        const auto type = mlir::RankedTensorType::get(shape, elementType);
        return FromOp(builder.create<mlir::bufferization::AllocTensorOp>(loc, type, sizes));
    }
};

mlir::ModuleOp ConvertASTToIR(mlir::MLIRContext& context, const ast::Module& node) {
    StencilIRGenerator generator{ context };

    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithmeticDialect>();
    context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    context.getOrLoadDialect<stencil::StencilDialect>();
    context.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
    context.getOrLoadDialect<mlir::tensor::TensorDialect>();

    auto [moduleOp, _] = generator.Generate(node);
    return mlir::dyn_cast<mlir::ModuleOp>(moduleOp);
}