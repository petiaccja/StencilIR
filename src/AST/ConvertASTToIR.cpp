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

static mlir::Type ConvertType(mlir::OpBuilder& builder, types::Type type, const TypeConversionOptions& options = {}) {
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
    if (lhsType == rhsType) {
        return { lhs, rhs };
    }
    if (auto promotedLhs = PromoteValue(builder, loc, lhs, rhsType)) {
        return { promotedLhs, rhs };
    }
    if (auto promotedRhs = PromoteValue(builder, loc, rhs, lhsType)) {
        return { lhs, promotedRhs };
    }
    throw std::logic_error("The two types don't have a common type.");
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
                                              ast::Print,
                                              ast::BinaryArithmeticOperator,
                                              ast::Constant<bool>,
                                              ast::Constant<float>,
                                              ast::Constant<double>,
                                              ast::Constant<int8_t>,
                                              ast::Constant<int16_t>,
                                              ast::Constant<int32_t>,
                                              ast::Constant<int64_t>,
                                              ast::Constant<uint8_t>,
                                              ast::Constant<uint16_t>,
                                              ast::Constant<uint32_t>,
                                              ast::Constant<uint64_t>> {
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
                         const std::vector<types::Type>& results,
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


    auto Generate(const ast::Module& node) const -> GenerationResult {
        auto op = builder.create<mlir::ModuleOp>(ConvertLocation(builder, node.location));

        auto loc = ConvertLocation(builder, node.location);
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
            for (int i = 0; i < op.getNumIterOperands(); ++i) {
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
        auto op = builder.create<mlir::scf::IfOp>(loc, condition, !node.bodyFalse.empty());

        auto& thenBlock = *op.thenBlock();
        thenBlock.clear();
        symbolTable.RunInScope([&] {
            InsertInBlock(thenBlock, [&] {
                for (auto& statement : node.bodyTrue) {
                    Generate(*statement);
                }
            });
        });

        auto& elseBlock = *op.elseBlock();
        elseBlock.clear();
        symbolTable.RunInScope([&] {
            InsertInBlock(elseBlock, [&] {
                for (auto& statement : node.bodyFalse) {
                    Generate(*statement);
                }
            });
        });

        throw std::runtime_error("Not implemented. Have to figure out result types somehow.");

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

    template <class T>
    auto Generate(const ast::Constant<T>& node) const -> GenerationResult {
        const auto loc = ConvertLocation(builder, node.location);
        if constexpr (std::is_floating_point_v<T>) {
            const auto type = std::is_same_v<T, float> ? builder.getF32Type() : builder.getF64Type();
            auto op = builder.create<mlir::arith::ConstantFloatOp>(loc,
                                                                   mlir::APFloat(node.value),
                                                                   type);
            return { op };
        }
        else if constexpr (std::is_same_v<T, bool>) {
            auto value = builder.getBoolAttr(node.value);
            auto op = builder.create<mlir::arith::ConstantOp>(loc, value, builder.getI1Type());
            return { op };
        }
        else if constexpr (std::is_integral_v<T>) {
            constexpr int numBits = sizeof(T) * 8;
            const auto type = node.type ? ConvertType(builder, node.type.value()) : mlir::Type(builder.getIntegerType(numBits));
            using Unsigned = std::make_unsigned_t<decltype(node.value)>;
            const uint64_t unsignedValue = std::bit_cast<Unsigned>(node.value);
            auto op = builder.create<mlir::arith::ConstantOp>(loc,
                                                              builder.getIntegerAttr(type, std::bit_cast<int64_t>(unsignedValue)),
                                                              type);
            return { op };
        }
        throw std::invalid_argument("Cannot lower this constant type.");
    }

    auto Generate(const ast::Print& node) const -> GenerationResult {
        const mlir::Value argument = Generate(*node.argument);

        auto op = builder.create<stencil::PrintOp>(ConvertLocation(builder, node.location),
                                                   argument);
        return { op };
    }

    auto Generate(const ast::BinaryArithmeticOperator& node) const -> GenerationResult {
        const auto loc = ConvertLocation(builder, node.location);
        auto [lhs, rhs] = PromoteToCommonType(builder, loc, Generate(*node.lhs), Generate(*node.rhs));
        bool isFloat = lhs.getType().isa<mlir::FloatType>();
        bool isUnsigned = lhs.getType().isUnsignedInteger();
        switch (node.operation) {
            case ast::BinaryArithmeticOperator::ADD:
                return { isFloat ? builder.create<mlir::arith::AddFOp>(loc, lhs, rhs)
                                 : builder.create<mlir::arith::AddIOp>(loc, lhs, rhs) };
            case ast::BinaryArithmeticOperator::SUB:
                return { isFloat ? builder.create<mlir::arith::SubFOp>(loc, lhs, rhs)
                                 : builder.create<mlir::arith::SubIOp>(loc, lhs, rhs) };
            case ast::BinaryArithmeticOperator::MUL:
                return { isFloat ? builder.create<mlir::arith::MulFOp>(loc, lhs, rhs)
                                 : builder.create<mlir::arith::MulIOp>(loc, lhs, rhs) };
            case ast::BinaryArithmeticOperator::DIV:
                return { isFloat      ? builder.create<mlir::arith::DivFOp>(loc, lhs, rhs)
                         : isUnsigned ? builder.create<mlir::arith::DivUIOp>(loc, lhs, rhs)
                                      : builder.create<mlir::arith::DivSIOp>(loc, lhs, rhs) };
            case ast::BinaryArithmeticOperator::MOD:
                return { isFloat      ? builder.create<mlir::arith::RemFOp>(loc, lhs, rhs)
                         : isUnsigned ? builder.create<mlir::arith::RemUIOp>(loc, lhs, rhs)
                                      : builder.create<mlir::arith::RemSIOp>(loc, lhs, rhs) };
            case ast::BinaryArithmeticOperator::BIT_AND:
                return { builder.create<mlir::arith::AndIOp>(loc, lhs, rhs) };
            case ast::BinaryArithmeticOperator::BIT_OR:
                return { builder.create<mlir::arith::OrIOp>(loc, lhs, rhs) };
            case ast::BinaryArithmeticOperator::BIT_XOR:
                return { builder.create<mlir::arith::XOrIOp>(loc, lhs, rhs) };
            case ast::BinaryArithmeticOperator::BIT_SHL:
                return { builder.create<mlir::arith::ShLIOp>(loc, lhs, rhs) };
            case ast::BinaryArithmeticOperator::BIT_SHR:
                return { isUnsigned ? builder.create<mlir::arith::ShRUIOp>(loc, lhs, rhs)
                                    : builder.create<mlir::arith::ShRSIOp>(loc, lhs, rhs) };
            default:
                throw std::logic_error("Binary op not implemented.");
        }
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