#include "ConvertASTToIR.hpp"

#include "IRGenerator.hpp"
#include "Nodes.hpp"
#include "SymbolTable.hpp"
#include "Types.hpp"

#include <Dialect/Stencil/IR/StencilOps.hpp>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
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

static mlir::Location ConvertLocation(mlir::OpBuilder& builder, const std::optional<ast::Location>& location) {
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

static mlir::Type ConvertType(mlir::OpBuilder& builder, ast::Type type, const TypeConversionOptions& options = {}) {
    struct {
        mlir::OpBuilder& builder;
        const TypeConversionOptions& options;
        mlir::Type operator()(const ast::ScalarType& type) const {
            switch (type) {
                case ast::ScalarType::SINT8: return builder.getIntegerType(8);
                case ast::ScalarType::SINT16: return builder.getIntegerType(16);
                case ast::ScalarType::SINT32: return builder.getIntegerType(32);
                case ast::ScalarType::SINT64: return builder.getIntegerType(64);
                // case ast::ScalarType::UINT8: return builder.getIntegerType(8, false);
                // case ast::ScalarType::UINT16: return builder.getIntegerType(16, false);
                // case ast::ScalarType::UINT32: return builder.getIntegerType(32, false);
                // case ast::ScalarType::UINT64: return builder.getIntegerType(64, false);
                case ast::ScalarType::UINT8: [[fallthrough]];
                case ast::ScalarType::UINT16: [[fallthrough]];
                case ast::ScalarType::UINT32: [[fallthrough]];
                case ast::ScalarType::UINT64:
                    assert(false && "Not supported due to stupid arith.constant");
                    std::terminate();
                case ast::ScalarType::INDEX: return builder.getIndexType();
                case ast::ScalarType::FLOAT32: return builder.getF32Type();
                case ast::ScalarType::FLOAT64: return builder.getF64Type();
                case ast::ScalarType::BOOL: return builder.getI1Type();
            }
            throw std::invalid_argument("Unknown type.");
        }
        mlir::Type operator()(const ast::FieldType& type) const {
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


static mlir::Value PromoteValue(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value value, mlir::Type type) {
    auto inputType = value.getType();
    if (inputType.isa<mlir::IntegerType>() && type.isa<mlir::IntegerType>()) {
        auto inputIntType = inputType.dyn_cast<mlir::IntegerType>();
        auto intType = type.dyn_cast<mlir::IntegerType>();
        if (inputIntType.getSignedness() == intType.getSignedness()) {
            if (inputIntType.getWidth() == intType.getWidth()) {
                return value;
            }
            if (inputIntType.getWidth() < intType.getWidth()) {
                const auto signedness = inputIntType.getSignedness();
                if (signedness == mlir::IntegerType::Unsigned) {
                    return builder.create<mlir::arith::ExtUIOp>(loc, type, value);
                }
                return builder.create<mlir::arith::ExtSIOp>(loc, type, value);
            }
        }
        return nullptr;
    }
    if (inputType.isa<mlir::FloatType>() && type.isa<mlir::FloatType>()) {
        auto inputFloatType = inputType.dyn_cast<mlir::FloatType>();
        auto floatType = type.dyn_cast<mlir::FloatType>();
        if (inputFloatType.getWidth() == floatType.getWidth()) {
            return value;
        }
        if (inputFloatType.getWidth() < floatType.getWidth()) {
            return builder.create<mlir::arith::ExtFOp>(loc, type, value);
        }
        return nullptr;
    }
    if (inputType.isa<mlir::IntegerType>() && type.isa<mlir::FloatType>()) {
        auto inputIntType = inputType.dyn_cast<mlir::IntegerType>();
        auto floatType = type.dyn_cast<mlir::FloatType>();
        if (inputIntType.getWidth() < floatType.getFPMantissaWidth()) {
            if (inputIntType.getSignedness() == mlir::IntegerType::Unsigned) {
                return builder.create<mlir::arith::UIToFPOp>(loc, floatType, value);
            }
            return builder.create<mlir::arith::SIToFPOp>(loc, floatType, value);
        }
        return nullptr;
    }
    return nullptr;
}


static std::pair<mlir::Value, mlir::Value> PromoteToCommonType(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) {
    const auto lhsType = lhs.getType();
    const auto rhsType = rhs.getType();
    std::string str;
    llvm::raw_string_ostream os(str);
    os << "The following types don't have a common type: ";
    lhsType.print(os);
    os << ", ";
    rhsType.print(os);
    os << ".";
    if (lhsType == rhsType) {
        return { lhs, rhs };
    }
    if (auto promotedLhs = PromoteValue(builder, loc, lhs, rhsType)) {
        return { promotedLhs, rhs };
    }
    if (auto promotedRhs = PromoteValue(builder, loc, rhs, lhsType)) {
        return { lhs, promotedRhs };
    }
    throw std::logic_error(std::move(str));
}



//------------------------------------------------------------------------------
// Generator
//------------------------------------------------------------------------------

struct GenerationResult {
    GenerationResult() = default;
    GenerationResult(mlir::Operation* op) : op(op), values{ op->getResults() } {}
    GenerationResult(mlir::SmallVector<mlir::Value> values) : values(values) {}
    operator mlir::Value() const {
        if (values.size() == 1) {
            return values.front();
        }
        throw std::logic_error("Result has multiple values when 1 value was requested");
    }
    mlir::Operation* op = nullptr;
    mlir::SmallVector<mlir::Value> values;
};

class StencilIRGenerator : public IRGenerator<ast::Node, GenerationResult, StencilIRGenerator,
                                              ast::Module,
                                              ast::Function,
                                              ast::Call,
                                              ast::Stencil,
                                              ast::Return,
                                              ast::Apply,
                                              ast::SymbolRef,
                                              ast::Assign,
                                              ast::Index,
                                              ast::Jump,
                                              ast::Sample,
                                              ast::JumpIndirect,
                                              ast::For,
                                              ast::If,
                                              ast::Yield,
                                              ast::SampleIndirect,
                                              ast::AllocTensor,
                                              ast::Dim,
                                              ast::ExtractSlice,
                                              ast::InsertSlice,
                                              ast::Print,
                                              ast::ArithmeticOperator,
                                              ast::ComparisonOperator,
                                              ast::Constant,
                                              ast::Cast> {
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

    auto GetFunctionType(const std::vector<ast::Parameter>& inputs,
                         const std::vector<ast::Type>& results,
                         TypeConversionOptions typeOptions) const {
        std::vector<mlir::Type> inputTypes{};
        for (auto& param : inputs) {
            inputTypes.push_back(ConvertType(builder, param.type, typeOptions));
        }
        std::vector<mlir::Type> resultTypes{};
        for (auto& result : results) {
            resultTypes.push_back(ConvertType(builder, result, typeOptions));
        }
        return builder.getFunctionType(mlir::TypeRange{ inputTypes }, mlir::TypeRange{ resultTypes });
    };

    auto GenerateFunctionBody(mlir::FunctionOpInterface op,
                              const std::vector<ast::Parameter>& parameters,
                              const std::vector<std::shared_ptr<ast::Statement>>& body) const {
        symbolTable.RunInScope([&, this] {
            auto& entryBlock = *op.addEntryBlock();

            for (size_t i = 0; i < parameters.size(); ++i) {
                symbolTable.Assign(parameters[i].name, entryBlock.getArgument(i));
            }

            InsertInBlock(entryBlock, [&] {
                for (const auto& statement : body) {
                    Generate(*statement);
                }
            });
        },
                               std::any((mlir::Operation*)op));
    }

    //--------------------------------------------------------------------------
    // Symbols
    //--------------------------------------------------------------------------
    auto Generate(const ast::SymbolRef& node) const -> GenerationResult {
        const auto value = symbolTable.Lookup(node.name);
        if (!value) {
            throw std::invalid_argument("Undefined symbol: " + node.name);
        }
        return { { *value } };
    }

    auto Generate(const ast::Assign& node) const -> GenerationResult {
        const auto ir = Generate(*node.expr);
        if (ir.values.size() != node.names.size()) {
            throw std::invalid_argument("Assign must have the same number of names as values.");
        }
        for (size_t i = 0; i < ir.values.size(); ++i) {
            symbolTable.Assign(node.names[i], ir.values[i]);
        }
        return {};
    }

    //--------------------------------------------------------------------------
    // Stencil structure
    //--------------------------------------------------------------------------

    auto Generate(const ast::Stencil& node) const -> GenerationResult {
        const TypeConversionOptions typeOptions = {
            .bufferType = TypeConversionOptions::TENSOR
        };

        const auto functionType = GetFunctionType(node.parameters, node.results, typeOptions);
        const auto loc = ConvertLocation(builder, node.location);
        auto op = builder.create<stencil::StencilOp>(loc,
                                                     node.name,
                                                     functionType,
                                                     mlir::APInt(64, node.numDimensions));

        GenerateFunctionBody(op, node.parameters, node.body);
        op.setVisibility(mlir::SymbolTable::Visibility::Private);

        return { op };
    }

    auto Generate(const ast::Apply& node) const -> GenerationResult {
        const auto callee = mlir::StringRef(node.callee);

        std::vector<mlir::Value> outputs;
        for (auto& target : node.outputs) {
            outputs.push_back(Generate(*target));
        }

        std::vector<mlir::Value> inputs;
        for (auto& argument : node.inputs) {
            inputs.push_back(Generate(*argument));
        }

        std::vector<mlir::Value> offsets;
        for (auto& offset : node.offsets) {
            offsets.push_back(Generate(*offset));
        }

        auto op = builder.create<stencil::ApplyOp>(ConvertLocation(builder, node.location),
                                                   callee,
                                                   inputs,
                                                   outputs,
                                                   offsets,
                                                   node.static_offsets);

        return { op };
    }

    auto Generate(const ast::Return& node) const -> GenerationResult {
        const auto loc = ConvertLocation(builder, node.location);
        std::vector<mlir::Value> values;
        for (const auto& value : node.values) {
            values.push_back(Generate(*value));
        }

        mlir::Operation* parentOp = std::any_cast<mlir::Operation*>(symbolTable.Info());

        if (mlir::isa<mlir::func::FuncOp>(parentOp)) {
            auto op = builder.create<mlir::func::ReturnOp>(loc, values);
            return { op };
        }
        else if (mlir::isa<stencil::StencilOp>(parentOp)) {
            auto op = builder.create<stencil::ReturnOp>(loc, values);
            return { op };
        }
        throw std::invalid_argument("ReturnOp must have either FuncOp or StencilOp as parent.");
    }

    //--------------------------------------------------------------------------
    // Module structure
    //--------------------------------------------------------------------------

    auto Generate(const ast::Function& node) const -> GenerationResult {
        const TypeConversionOptions typeOptions = {
            .bufferType = TypeConversionOptions::TENSOR
        };

        const auto functionType = GetFunctionType(node.parameters, node.results, typeOptions);
        const auto loc = ConvertLocation(builder, node.location);
        auto op = builder.create<mlir::func::FuncOp>(loc,
                                                     node.name,
                                                     functionType);

        GenerateFunctionBody(op, node.parameters, node.body);

        return { op };
    }

    auto Generate(const ast::Call& node) const -> GenerationResult {
        const auto loc = ConvertLocation(builder, node.location);
        const auto parentOp = builder.getBlock()->getParentOp();
        const auto calleeAttr = builder.getStringAttr(node.callee);
        const auto calleeOp = mlir::SymbolTable::lookupNearestSymbolFrom(parentOp, calleeAttr);
        const auto calleeFuncOp = mlir::dyn_cast<mlir::func::FuncOp>(calleeOp);
        if (!calleeFuncOp) {
            throw std::invalid_argument("Function not found: " + node.callee);
        }
        mlir::SmallVector<mlir::Value, 8> args;
        for (auto& arg : node.args) {
            args.push_back(Generate(*arg));
        }
        return { builder.create<mlir::func::CallOp>(loc, calleeFuncOp, args) };
    }

    auto Generate(const ast::Module& node) const -> GenerationResult {
        auto op = builder.create<mlir::ModuleOp>(ConvertLocation(builder, node.location));

        builder.setInsertionPointToEnd(op.getBody());

        for (auto& kernel : node.stencils) {
            Generate(*kernel);
        }
        for (auto& function : node.functions) {
            Generate(*function);
        }

        return { op };
    }

    //--------------------------------------------------------------------------
    // Stencil intrinsics
    //--------------------------------------------------------------------------

    auto Generate(const ast::Index& node) const -> GenerationResult {
        const auto loc = ConvertLocation(builder, node.location);

        stencil::StencilOp currentStencil = nullptr;
        const auto& scopeInfo = symbolTable.Info();
        if (scopeInfo.has_value() && scopeInfo.type() == typeid(mlir::Operation*)) {
            const auto op = std::any_cast<mlir::Operation*>(scopeInfo);
            currentStencil = mlir::dyn_cast<stencil::StencilOp>(op);
        }
        if (!currentStencil) {
            throw std::invalid_argument("IndexOp can only be used inside StencilOps");
        }
        const int64_t numDims = currentStencil.getNumDimensions().getSExtValue();
        const auto indexType = mlir::VectorType::get({ numDims }, builder.getIndexType());

        auto op = builder.create<stencil::IndexOp>(loc, indexType);
        return { op };
    }

    auto Generate(const ast::Jump& node) const -> GenerationResult {
        const auto loc = ConvertLocation(builder, node.location);
        const mlir::Value index = Generate(*node.index);
        const auto offset = builder.getI64ArrayAttr(node.offset);
        auto op = builder.create<stencil::JumpOp>(loc, index.getType(), index, offset);
        return { op };
    }

    auto Generate(const ast::Sample& node) const -> GenerationResult {
        const auto loc = ConvertLocation(builder, node.location);
        const mlir::Value field = Generate(*node.field);
        const mlir::Value index = Generate(*node.index);

        auto fieldType = field.getType();
        if (!fieldType.isa<mlir::ShapedType>()) {
            throw std::invalid_argument("SampleOp must be used to sample fields.");
        }
        auto elementType = fieldType.dyn_cast<mlir::ShapedType>().getElementType();

        auto op = builder.create<stencil::SampleOp>(loc, elementType, field, index);
        return { op };
    }

    auto Generate(const ast::JumpIndirect& node) const -> GenerationResult {
        const auto loc = ConvertLocation(builder, node.location);

        const mlir::Value index = Generate(*node.index);
        const auto dimension = builder.getIndexAttr(node.dimension);
        const mlir::Value map = Generate(*node.map);
        const mlir::Value mapElement = Generate(*node.mapElement);

        auto op = builder.create<stencil::JumpIndirectOp>(loc, index.getType(), index, dimension, map, mapElement);
        return { op };
    }

    auto Generate(const ast::SampleIndirect& node) const -> GenerationResult {
        const auto loc = ConvertLocation(builder, node.location);
        const mlir::Value index = Generate(*node.index);
        const auto dimension = builder.getIndexAttr(node.dimension);
        const mlir::Value field = Generate(*node.field);
        const mlir::Value fieldElement = Generate(*node.fieldElement);

        auto fieldType = field.getType();
        if (!fieldType.isa<mlir::ShapedType>()) {
            throw std::invalid_argument("SampleIndirectOp must be used to sample fields.");
        }
        auto elementType = fieldType.dyn_cast<mlir::ShapedType>().getElementType();

        auto op = builder.create<stencil::SampleIndirectOp>(loc, elementType, index, dimension, field, fieldElement);
        return { op };
    }

    //--------------------------------------------------------------------------
    // Structured flow control
    //--------------------------------------------------------------------------

    auto Generate(const ast::For& node) const -> GenerationResult {
        const auto loc = ConvertLocation(builder, node.location);

        if (node.initArgs.size() != node.iterArgSymbols.size()) {
            throw std::invalid_argument("for loop must have the same number of init args as iter args");
        }

        const mlir::Value start = Generate(*node.start);
        const mlir::Value end = Generate(*node.end);
        const mlir::Value step = builder.create<mlir::arith::ConstantIndexOp>(loc, node.step);
        mlir::SmallVector<mlir::Value, 4> initArgs;
        for (auto& v : node.initArgs) {
            initArgs.push_back(Generate(*v));
        }

        auto op = builder.create<mlir::scf::ForOp>(loc, start, end, step, initArgs);

        auto& body = *op.getBody();
        body.clear();
        symbolTable.RunInScope([&, this] {
            symbolTable.Assign(node.loopVarSymbol, op.getInductionVar());
            for (size_t i = 0; i < op.getNumIterOperands(); ++i) {
                symbolTable.Assign(node.iterArgSymbols[i], op.getRegionIterArgs()[i]);
            }
            InsertInBlock(body, [&]() {
                for (auto& statement : node.body) {
                    Generate(*statement);
                }
            });
        },
                               std::any{ (mlir::Operation*)op });

        return { op };
    }

    auto Generate(const ast::If& node) const -> GenerationResult {
        const auto loc = ConvertLocation(builder, node.location);

        const mlir::Value condition = Generate(*node.condition);

        // Create an IfOp without results.
        auto deductionOp = builder.create<mlir::scf::IfOp>(loc, condition, false);

        // Fill in the IfOp's then block to deduce the yielded result types.
        auto& deductionBlock = *deductionOp.thenBlock();
        deductionBlock.clear();
        symbolTable.RunInScope([&] {
            InsertInBlock(deductionBlock, [&] {
                for (const auto& statement : node.thenBody) {
                    // We only need to generate Assign and Yield ops.
                    // All the rest can't provide an argument to Yield.
                    const ast::Statement& stmt = *statement;
                    if (typeid(stmt) == typeid(ast::Assign)
                        || typeid(stmt) == typeid(ast::Yield)) {
                        Generate(*statement);
                    }
                }
            });
        });

        // Extract result types from deduction op.
        mlir::SmallVector<mlir::Type, 4> resultTypes;
        if (!deductionOp.thenBlock()->empty()) {
            const auto resultTypesView = deductionOp.thenYield()->getOperandTypes();
            resultTypes = { resultTypesView.begin(), resultTypesView.end() };
        }
        deductionOp->erase();

        // Create the actual IfOp with result types and both blocks.
        const bool hasElseBlock = !node.elseBody.empty();
        auto op = builder.create<mlir::scf::IfOp>(loc, resultTypes, condition, hasElseBlock);

        auto& thenBlock = *op.thenBlock();
        thenBlock.clear();
        symbolTable.RunInScope([&] {
            InsertInBlock(thenBlock, [&] {
                for (const auto& statement : node.thenBody) {
                    Generate(*statement);
                }
            });
        });

        if (hasElseBlock) {
            auto& elseBlock = *op.elseBlock();
            elseBlock.clear();
            symbolTable.RunInScope([&] {
                InsertInBlock(elseBlock, [&] {
                    for (const auto& statement : node.elseBody) {
                        Generate(*statement);
                    }
                });
            });
        }

        // Return the actual IfOp, not the deduction which has been erased.
        return { op };
    }

    auto Generate(const ast::Yield& node) const -> GenerationResult {
        const auto loc = ConvertLocation(builder, node.location);
        std::vector<mlir::Value> values;
        for (const auto& value : node.values) {
            values.push_back(Generate(*value));
        }

        auto op = builder.create<mlir::scf::YieldOp>(loc, values);
        return { op };
    }

    //--------------------------------------------------------------------------
    // Arithmetic and misc
    //--------------------------------------------------------------------------

    auto Generate(const ast::Constant& node) const -> GenerationResult {
        const auto loc = ConvertLocation(builder, node.location);
        const auto type = ConvertType(builder, node.type);

        auto op = ast::VisitType(node.type, [&](auto* t) -> mlir::Operation* {
            using T = std::decay_t<std::remove_pointer_t<decltype(t)>>;
            const auto value = std::any_cast<T>(node.value);

            if (node.type == ast::ScalarType::INDEX) {
                return builder.create<mlir::arith::ConstantIndexOp>(loc, value);
            }
            if (node.type == ast::ScalarType::BOOL) {
                return builder.create<mlir::arith::ConstantIntOp>(loc, int64_t(value), type);
            }
            if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
                auto signlessType = builder.getIntegerType(type.dyn_cast<mlir::IntegerType>().getWidth());
                return builder.create<mlir::arith::ConstantIntOp>(loc, value, signlessType);
            }
            if constexpr (std::is_integral_v<T> && std::is_unsigned_v<T>) {
                auto signlessType = builder.getIntegerType(type.dyn_cast<mlir::IntegerType>().getWidth());
                const int64_t equivalent = std::bit_cast<int64_t>(uint64_t(value));
                return builder.create<mlir::arith::ConstantIntOp>(loc, equivalent, signlessType);
            }
            if constexpr (std::is_floating_point_v<T>) {
                return builder.create<mlir::arith::ConstantFloatOp>(loc, mlir::APFloat(value), type.dyn_cast<mlir::FloatType>());
            }
            throw std::invalid_argument("Invalid type.");
        });
        return { op };
    }

    auto Generate(const ast::Print& node) const -> GenerationResult {
        const mlir::Value argument = Generate(*node.argument);

        auto op = builder.create<stencil::PrintOp>(ConvertLocation(builder, node.location),
                                                   argument);
        return { op };
    }

    auto Generate(const ast::ArithmeticOperator& node) const -> GenerationResult {
        const auto loc = ConvertLocation(builder, node.location);
        auto [lhs, rhs] = PromoteToCommonType(builder, loc, Generate(*node.lhs), Generate(*node.rhs));
        bool isFloat = lhs.getType().isa<mlir::FloatType>();
        bool isUnsigned = lhs.getType().isUnsignedInteger();
        switch (node.operation) {
            case ast::eArithmeticFunction::ADD:
                return { isFloat ? builder.create<mlir::arith::AddFOp>(loc, lhs, rhs)
                                 : builder.create<mlir::arith::AddIOp>(loc, lhs, rhs) };
            case ast::eArithmeticFunction::SUB:
                return { isFloat ? builder.create<mlir::arith::SubFOp>(loc, lhs, rhs)
                                 : builder.create<mlir::arith::SubIOp>(loc, lhs, rhs) };
            case ast::eArithmeticFunction::MUL:
                return { isFloat ? builder.create<mlir::arith::MulFOp>(loc, lhs, rhs)
                                 : builder.create<mlir::arith::MulIOp>(loc, lhs, rhs) };
            case ast::eArithmeticFunction::DIV:
                return { isFloat      ? builder.create<mlir::arith::DivFOp>(loc, lhs, rhs)
                         : isUnsigned ? builder.create<mlir::arith::DivUIOp>(loc, lhs, rhs)
                                      : builder.create<mlir::arith::DivSIOp>(loc, lhs, rhs) };
            case ast::eArithmeticFunction::MOD:
                return { isFloat      ? builder.create<mlir::arith::RemFOp>(loc, lhs, rhs)
                         : isUnsigned ? builder.create<mlir::arith::RemUIOp>(loc, lhs, rhs)
                                      : builder.create<mlir::arith::RemSIOp>(loc, lhs, rhs) };
            case ast::eArithmeticFunction::BIT_AND:
                return { builder.create<mlir::arith::AndIOp>(loc, lhs, rhs) };
            case ast::eArithmeticFunction::BIT_OR:
                return { builder.create<mlir::arith::OrIOp>(loc, lhs, rhs) };
            case ast::eArithmeticFunction::BIT_XOR:
                return { builder.create<mlir::arith::XOrIOp>(loc, lhs, rhs) };
            case ast::eArithmeticFunction::BIT_SHL:
                return { builder.create<mlir::arith::ShLIOp>(loc, lhs, rhs) };
            case ast::eArithmeticFunction::BIT_SHR:
                return { isUnsigned ? builder.create<mlir::arith::ShRUIOp>(loc, lhs, rhs)
                                    : builder.create<mlir::arith::ShRSIOp>(loc, lhs, rhs) };
        }
        throw std::logic_error("Binary op not implemented.");
    }

    auto Generate(const ast::ComparisonOperator& node) const -> GenerationResult {
        const auto loc = ConvertLocation(builder, node.location);
        auto [lhs, rhs] = PromoteToCommonType(builder, loc, Generate(*node.lhs), Generate(*node.rhs));
        bool isFloat = lhs.getType().isa<mlir::FloatType>();
        bool isUnsigned = lhs.getType().isUnsignedInteger();
        switch (node.operation) {
            case ast::eComparisonFunction::EQ:
                return { isFloat ? builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ, lhs, rhs)
                                 : builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, lhs, rhs) };
            case ast::eComparisonFunction::NEQ:
                return { isFloat ? builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::ONE, lhs, rhs)
                                 : builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, lhs, rhs) };
            case ast::eComparisonFunction::GT:
                return { isFloat      ? builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, lhs, rhs)
                         : isUnsigned ? builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ugt, lhs, rhs)
                                      : builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sgt, lhs, rhs) };
            case ast::eComparisonFunction::LT:
                return { isFloat      ? builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, lhs, rhs)
                         : isUnsigned ? builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, lhs, rhs)
                                      : builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, lhs, rhs) };
            case ast::eComparisonFunction::GTE:
                return { isFloat      ? builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGE, lhs, rhs)
                         : isUnsigned ? builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::uge, lhs, rhs)
                                      : builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, lhs, rhs) };
            case ast::eComparisonFunction::LTE:
                return { isFloat      ? builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLE, lhs, rhs)
                         : isUnsigned ? builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ule, lhs, rhs)
                                      : builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sle, lhs, rhs) };
        }
        throw std::logic_error("Binary op not implemented.");
    }

    auto Generate(const ast::Cast& node) const -> GenerationResult {
        auto loc = ConvertLocation(builder, node.location);
        mlir::Value expr = Generate(*node.expr);
        mlir::Type type = ConvertType(builder, node.type);

        auto ftype = type.dyn_cast<mlir::FloatType>();
        auto fexpr = expr.getType().dyn_cast<mlir::FloatType>();
        auto itype = type.dyn_cast<mlir::IntegerType>();
        auto iexpr = expr.getType().dyn_cast<mlir::IntegerType>();

        if (type.isa<mlir::IndexType>() || expr.getType().isa<mlir::IndexType>()) {
            if (iexpr) {
                mlir::Type signlessType = builder.getIntegerType(iexpr.getWidth());
                mlir::Value signlessExpr = builder.create<mlir::arith::BitcastOp>(loc, signlessType, expr);
                return { builder.create<mlir::arith::IndexCastOp>(loc, type, signlessExpr) };
            }
            return { builder.create<mlir::arith::IndexCastOp>(loc, type, expr) };
        }
        if (ftype && fexpr) {
            if (ftype.getWidth() > fexpr.getWidth()) {
                return { builder.create<mlir::arith::ExtFOp>(loc, type, expr) };
            }
            else if (ftype.getWidth() < fexpr.getWidth()) {
                return { builder.create<mlir::arith::TruncFOp>(loc, type, expr) };
            }
            else {
                return { { expr } };
            }
        }
        if (itype && iexpr) {
            bool isSigned = !(itype.isUnsigned() && iexpr.isUnsigned());
            if (ftype.getWidth() > fexpr.getWidth()) {
                return isSigned ? GenerationResult{ builder.create<mlir::arith::ExtSIOp>(loc, type, expr) }
                                : GenerationResult{ builder.create<mlir::arith::ExtUIOp>(loc, type, expr) };
            }
            else if (ftype.getWidth() < fexpr.getWidth()) {
                return { builder.create<mlir::arith::TruncIOp>(loc, type, expr) };
            }
            else {
                return { { expr } };
            }
        }
        if (itype && fexpr) {
            return itype.isUnsigned() ? GenerationResult{ builder.create<mlir::arith::FPToUIOp>(loc, type, expr) }
                                      : GenerationResult{ builder.create<mlir::arith::FPToSIOp>(loc, type, expr) };
        }
        if (ftype && iexpr) {
            return iexpr.isUnsigned() ? GenerationResult{ builder.create<mlir::arith::UIToFPOp>(loc, type, expr) }
                                      : GenerationResult{ builder.create<mlir::arith::SIToFPOp>(loc, type, expr) };
        }
        throw std::invalid_argument("No conversion implemented between given types.");
    }


    //--------------------------------------------------------------------------
    // Tensor
    //--------------------------------------------------------------------------

    auto Generate(const ast::Dim& node) const -> GenerationResult {
        auto loc = ConvertLocation(builder, node.location);
        const mlir::Value tensor = Generate(*node.field);
        const mlir::Value index = Generate(*node.index);
        if (tensor.getType().isa<mlir::MemRefType>()) {
            return { builder.create<mlir::memref::DimOp>(loc, tensor, index) };
        }
        else {
            return { builder.create<mlir::tensor::DimOp>(loc, tensor, index) };
        }
    }

    auto Generate(const ast::AllocTensor& node) const -> GenerationResult {
        auto loc = ConvertLocation(builder, node.location);

        const auto elementType = ConvertType(builder, node.elementType);
        std::vector<mlir::Value> sizes;
        std::vector<int64_t> shape;
        for (const auto& size : node.sizes) {
            sizes.push_back(Generate(*size));
            shape.push_back(mlir::ShapedType::kDynamicSize);
        }
        const auto type = mlir::RankedTensorType::get(shape, elementType);
        return { builder.create<mlir::bufferization::AllocTensorOp>(loc, type, sizes) };
    }

    auto Generate(const ast::ExtractSlice& node) const -> GenerationResult {
        auto loc = ConvertLocation(builder, node.location);

        const mlir::Value source = Generate(*node.source);
        std::vector<mlir::Value> offsets;
        std::vector<mlir::Value> sizes;
        std::vector<mlir::Value> strides;

        for (auto& offset : node.offsets) {
            offsets.push_back(Generate(*offset));
        }
        for (auto& size : node.sizes) {
            sizes.push_back(Generate(*size));
        }
        for (auto& stride : node.strides) {
            strides.push_back(Generate(*stride));
        }

        return { builder.create<mlir::tensor::ExtractSliceOp>(loc, source, offsets, sizes, strides) };
    }

    auto Generate(const ast::InsertSlice& node) const -> GenerationResult {
        auto loc = ConvertLocation(builder, node.location);

        const mlir::Value source = Generate(*node.source);
        const mlir::Value dest = Generate(*node.dest);
        std::vector<mlir::Value> offsets;
        std::vector<mlir::Value> sizes;
        std::vector<mlir::Value> strides;

        for (auto& offset : node.offsets) {
            offsets.push_back(Generate(*offset));
        }
        for (auto& size : node.sizes) {
            sizes.push_back(Generate(*size));
        }
        for (auto& stride : node.strides) {
            strides.push_back(Generate(*stride));
        }

        return { builder.create<mlir::tensor::InsertSliceOp>(loc, source, dest, offsets, sizes, strides) };
    }

private:
    mutable mlir::OpBuilder builder;
    mutable SymbolTable<std::string, mlir::Value> symbolTable;
};

mlir::ModuleOp ConvertASTToIR(mlir::MLIRContext& context, const ast::Module& node) {
    StencilIRGenerator generator{ context };

    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithmeticDialect>();
    context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    context.getOrLoadDialect<stencil::StencilDialect>();
    context.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
    context.getOrLoadDialect<mlir::tensor::TensorDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();

    auto ir = generator.Generate(node);
    auto module = mlir::dyn_cast<mlir::ModuleOp>(ir.op);
    if (failed(module.verify())) {
        throw std::logic_error("MLIR module is not correct.");
    }
    return module;
}