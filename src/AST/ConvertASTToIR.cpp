#include "ConvertASTToIR.hpp"

#include "IRGenerator.hpp"
#include "Nodes.hpp"
#include "SymbolTable.hpp"
#include "Types.hpp"

#include <Diagnostics/Exception.hpp>
#include <Diagnostics/Formatting.hpp>
#include <Diagnostics/Handlers.hpp>
#include <Dialect/Stencil/IR/StencilOps.hpp>
#include <Dialect/Stencil/Transforms/BufferizableOpInterfaceImpl.hpp>

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
#include <mlir/IR/Verifier.h>
#include <mlir/InitAllDialects.h>
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

static std::string FormatType(mlir::Type type) {
    std::string s;
    llvm::raw_string_ostream os{ s };
    type.print(os);
    return s;
}


static mlir::Type ConvertType(mlir::OpBuilder& builder, const ast::Type& type) {
    if (auto integerType = dynamic_cast<const ast::IntegerType*>(&type)) {
        if (!integerType->isSigned) {
            throw std::invalid_argument("unsigned types are not supported due to arith.constant behavior; TODO: add support");
        }
        return builder.getIntegerType(integerType->size);
    }
    else if (auto floatType = dynamic_cast<const ast::FloatType*>(&type)) {
        switch (floatType->size) {
            case 16: return builder.getF16Type();
            case 32: return builder.getF32Type();
            case 64: return builder.getF64Type();
        }
        throw std::invalid_argument("only 16, 32, and 64-bit floats are supported");
    }
    else if (auto indexType = dynamic_cast<const ast::IndexType*>(&type)) {
        return builder.getIndexType();
    }
    else if (auto fieldType = dynamic_cast<const ast::FieldType*>(&type)) {
        const mlir::Type elementType = ConvertType(builder, *fieldType->elementType);
        std::vector<int64_t> shape(fieldType->numDimensions, mlir::ShapedType::kDynamicSize);
        return mlir::RankedTensorType::get(shape, elementType);
    }
    else {
        std::stringstream ss;
        ss << "could not convert type \"" << type << "\" to MLIR type";
        throw std::invalid_argument(ss.str());
    }
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
    throw OperandTypeError(loc, { FormatType(lhsType), FormatType(rhsType) });
}


//------------------------------------------------------------------------------
// Generator
//------------------------------------------------------------------------------

struct GenerationResult {
    GenerationResult() = default;
    GenerationResult(mlir::Operation* op) : op(op), values{ op->getResults() } {}
    GenerationResult(mlir::SmallVector<mlir::Value> values) : values(values) { assert(!values.empty()); }
    operator mlir::Value() const {
        if (values.size() == 1) {
            return values.front();
        }
        mlir::Location location = op ? op->getLoc() : mlir::UnknownLoc::get(values.front().getContext());
        throw ArgumentCountError(location, 1, int(values.size()));
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
                                              ast::Pack,
                                              ast::Index,
                                              ast::Jump,
                                              ast::Project,
                                              ast::Extend,
                                              ast::Exchange,
                                              ast::Extract,
                                              ast::Sample,
                                              ast::For,
                                              ast::If,
                                              ast::Yield,
                                              ast::Block,
                                              ast::AllocTensor,
                                              ast::Dim,
                                              ast::ExtractSlice,
                                              ast::InsertSlice,
                                              ast::Print,
                                              ast::ArithmeticOperator,
                                              ast::ComparisonOperator,
                                              ast::Min,
                                              ast::Max,
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
                         const std::vector<ast::TypePtr>& results) const {
        std::vector<mlir::Type> inputTypes{};
        for (auto& param : inputs) {
            inputTypes.push_back(ConvertType(builder, *param.type));
        }
        std::vector<mlir::Type> resultTypes{};
        for (auto& result : results) {
            resultTypes.push_back(ConvertType(builder, *result));
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
            throw UndefinedSymbolError{ ConvertLocation(builder, node.location), node.name };
        }
        return { { *value } };
    }

    auto Generate(const ast::Assign& node) const -> GenerationResult {
        mlir::SmallVector<mlir::Value, 6> values;
        for (const auto& expr : node.exprs) {
            const auto generationResult = Generate(*expr);
            for (const auto& v : generationResult.values) {
                values.push_back(v);
            }
        }
        if (values.size() != node.names.size()) {
            throw ArgumentCountError{ ConvertLocation(builder, node.location), int(node.names.size()), int(values.size()) };
        }
        for (size_t i = 0; i < values.size(); ++i) {
            symbolTable.Assign(node.names[i], values[i]);
        }
        return {};
    }

    auto Generate(const ast::Pack& node) const -> GenerationResult {
        mlir::SmallVector<mlir::Value, 6> values;
        for (const auto& expr : node.exprs) {
            const auto generationResult = Generate(*expr);
            for (const auto& v : generationResult.values) {
                values.push_back(v);
            }
        }
        return { std::move(values) };
    }

    //--------------------------------------------------------------------------
    // Stencil structure
    //--------------------------------------------------------------------------

    auto Generate(const ast::Stencil& node) const -> GenerationResult {
        const auto functionType = GetFunctionType(node.parameters, node.results);
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
            const auto generationResult = Generate(*value);
            for (const auto& v : generationResult.values) {
                values.push_back(v);
            }
        }

        mlir::Operation* parentOp = std::any_cast<mlir::Operation*>(symbolTable.Info());

        if (mlir::isa<mlir::func::FuncOp>(parentOp)) {
            auto op = builder.create<mlir::func::ReturnOp>(loc, values);
            return { op };
        }
        else {
            auto op = builder.create<stencil::ReturnOp>(loc, values);
            return { op };
        }
    }

    //--------------------------------------------------------------------------
    // Module structure
    //--------------------------------------------------------------------------

    auto Generate(const ast::Function& node) const -> GenerationResult {
        const auto functionType = GetFunctionType(node.parameters, node.results);
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
            throw UndefinedSymbolError{ loc, node.callee };
        }
        mlir::SmallVector<mlir::Value, 8> args;
        for (auto& arg : node.args) {
            const auto generationResult = Generate(*arg);
            for (const auto& v : generationResult.values) {
                args.push_back(v);
            }
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

        auto currentBlock = builder.getBlock();
        auto currentOp = currentBlock->getParentOp();
        stencil::StencilOp currentStencil = mlir::dyn_cast<stencil::StencilOp>(currentOp)
                                                ? mlir::dyn_cast<stencil::StencilOp>(currentOp)
                                                : currentOp->getParentOfType<stencil::StencilOp>();
        if (!currentStencil) {
            auto msg = FormatDiagnostic(FormatLocation(loc),
                                        FormatSeverity(mlir::DiagnosticSeverity::Error),
                                        "index op can only be used within stencils");
            throw Exception(std::move(msg));
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

    auto Generate(const ast::Project& node) const -> GenerationResult {
        const auto loc = ConvertLocation(builder, node.location);
        const mlir::Value index = Generate(*node.index);
        auto op = builder.create<stencil::ProjectOp>(loc, index, node.positions);
        return { op };
    }

    auto Generate(const ast::Extend& node) const -> GenerationResult {
        const auto loc = ConvertLocation(builder, node.location);
        const mlir::Value index = Generate(*node.index);
        const mlir::Value value = Generate(*node.value);
        auto op = builder.create<stencil::ExtendOp>(loc, index, node.position, value);
        return { op };
    }

    auto Generate(const ast::Exchange& node) const -> GenerationResult {
        const auto loc = ConvertLocation(builder, node.location);
        const mlir::Value index = Generate(*node.index);
        const mlir::Value value = Generate(*node.value);
        auto op = builder.create<stencil::ExchangeOp>(loc, index, node.position, value);
        return { op };
    }

    auto Generate(const ast::Extract& node) const -> GenerationResult {
        const auto loc = ConvertLocation(builder, node.location);
        const mlir::Value index = Generate(*node.index);
        auto op = builder.create<stencil::ExtractOp>(loc, index, node.position);
        return { op };
    }

    auto Generate(const ast::Sample& node) const -> GenerationResult {
        const auto loc = ConvertLocation(builder, node.location);
        const mlir::Value field = Generate(*node.field);
        const mlir::Value index = Generate(*node.index);

        auto fieldType = field.getType();
        if (!fieldType.isa<mlir::ShapedType>()) {
            throw ArgumentTypeError{ loc, FormatType(fieldType), 0 };
        }
        auto elementType = fieldType.dyn_cast<mlir::ShapedType>().getElementType();

        auto op = builder.create<stencil::SampleOp>(loc, elementType, field, index);
        return { op };
    }

    //--------------------------------------------------------------------------
    // Structured flow control
    //--------------------------------------------------------------------------

    auto Generate(const ast::For& node) const -> GenerationResult {
        const auto loc = ConvertLocation(builder, node.location);

        const mlir::Value start = Generate(*node.start);
        const mlir::Value end = Generate(*node.end);
        const mlir::Value step = Generate(*node.step);
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

        auto insertionBlock = builder.getInsertionBlock();
        auto insertionPoint = builder.getInsertionPoint();

        // Create a block to infer result types.
        auto currentRegion = builder.getInsertionBlock()->getParent();
        auto& block = *builder.createBlock(currentRegion, currentRegion->end());
        symbolTable.RunInScope([&] {
            InsertInBlock(block, [&] {
                for (const auto& statement : node.thenBody) {
                    Generate(*statement);
                }
            });
        });
        mlir::SmallVector<mlir::Type, 4> resultTypes;
        if (!block.empty()) {
            const auto resultTypesView = block.getTerminator()->getOperandTypes();
            resultTypes = { resultTypesView.begin(), resultTypesView.end() };
        }

        builder.setInsertionPoint(insertionBlock, insertionPoint);

        // Create the actual IfOp with result types and both blocks.
        const bool hasElseBlock = !node.elseBody.empty();
        auto op = builder.create<mlir::scf::IfOp>(loc, resultTypes, condition, hasElseBlock);

        auto& thenBlock = *op.thenBlock();
        block.moveBefore(&thenBlock);
        thenBlock.erase();

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
            const auto generationResult = Generate(*value);
            for (const auto& v : generationResult.values) {
                values.push_back(v);
            }
        }

        auto op = builder.create<mlir::scf::YieldOp>(loc, values);
        return { op };
    }

    auto Generate(const ast::Block& node) const -> GenerationResult {
        const auto loc = ConvertLocation(builder, node.location);

        auto insertionBlock = builder.getInsertionBlock();
        auto insertionPoint = builder.getInsertionPoint();

        auto currentRegion = builder.getInsertionBlock()->getParent();
        auto& block = *builder.createBlock(currentRegion, currentRegion->end());
        symbolTable.RunInScope([&] {
            InsertInBlock(block, [&] {
                for (const auto& statement : node.body) {
                    Generate(*statement);
                }
            });
        });
        mlir::SmallVector<mlir::Type, 4> resultTypes;
        if (!block.empty()) {
            const auto resultTypesView = block.getTerminator()->getOperandTypes();
            resultTypes = { resultTypesView.begin(), resultTypesView.end() };
        }

        builder.setInsertionPoint(insertionBlock, insertionPoint);

        auto op = builder.create<mlir::scf::ExecuteRegionOp>(loc, resultTypes);
        auto& sentinelBlock = op.getRegion().emplaceBlock();
        block.moveBefore(&sentinelBlock);
        sentinelBlock.erase();
        return { op };
    }

    //--------------------------------------------------------------------------
    // Arithmetic and misc
    //--------------------------------------------------------------------------

    auto Generate(const ast::Constant& node) const -> GenerationResult {
        const auto loc = ConvertLocation(builder, node.location);
        const auto type = ConvertType(builder, *node.type);

        if (std::dynamic_pointer_cast<ast::IntegerType>(node.type)) {
            const int64_t value = std::any_cast<int64_t>(node.value);
            auto signlessType = builder.getIntegerType(type.dyn_cast<mlir::IntegerType>().getWidth());
            return { builder.create<mlir::arith::ConstantIntOp>(loc, value, signlessType) };
        }
        else if (std::dynamic_pointer_cast<ast::IndexType>(node.type)) {
            const int64_t value = std::any_cast<int64_t>(node.value);
            return { builder.create<mlir::arith::ConstantIndexOp>(loc, value) };
        }
        else if (auto floatType = std::dynamic_pointer_cast<ast::FloatType>(node.type)) {
            const double value = std::any_cast<double>(node.value);
            const auto apfloat = floatType->size == 32 ? mlir::APFloat(float(value)) : mlir::APFloat(double(value));
            return { builder.create<mlir::arith::ConstantFloatOp>(loc, apfloat, type.dyn_cast<mlir::FloatType>()) };
        }
        throw ArgumentTypeError{ loc, FormatType(type), 0 };
    }

    auto
    Generate(const ast::Print& node) const -> GenerationResult {
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
        throw NotImplementedError("switch does not cover operator");
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
        throw NotImplementedError("switch does not cover operator");
    }

    auto Generate(const ast::Min& node) const -> GenerationResult {
        const auto loc = ConvertLocation(builder, node.location);
        auto [lhs, rhs] = PromoteToCommonType(builder, loc, Generate(*node.lhs), Generate(*node.rhs));
        bool isFloat = lhs.getType().isa<mlir::FloatType>();
        bool isUnsigned = lhs.getType().isUnsignedInteger();

        return { isFloat      ? builder.create<mlir::arith::MinFOp>(loc, lhs, rhs)
                 : isUnsigned ? builder.create<mlir::arith::MinUIOp>(loc, lhs, rhs)
                              : builder.create<mlir::arith::MinSIOp>(loc, lhs, rhs) };
    }

    auto Generate(const ast::Max& node) const -> GenerationResult {
        const auto loc = ConvertLocation(builder, node.location);
        auto [lhs, rhs] = PromoteToCommonType(builder, loc, Generate(*node.lhs), Generate(*node.rhs));
        bool isFloat = lhs.getType().isa<mlir::FloatType>();
        bool isUnsigned = lhs.getType().isUnsignedInteger();

        return { isFloat      ? builder.create<mlir::arith::MaxFOp>(loc, lhs, rhs)
                 : isUnsigned ? builder.create<mlir::arith::MaxUIOp>(loc, lhs, rhs)
                              : builder.create<mlir::arith::MaxSIOp>(loc, lhs, rhs) };
    }

    auto Generate(const ast::Cast& node) const -> GenerationResult {
        auto loc = ConvertLocation(builder, node.location);
        mlir::Value expr = Generate(*node.expr);
        mlir::Type type = ConvertType(builder, *node.type);

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

        const auto elementType = ConvertType(builder, *node.elementType);
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
    mlir::registerAllDialects(context);
    mlir::DialectRegistry registry;
    registry.insert<stencil::StencilDialect>();
    mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
    stencil::registerBufferizableOpInterfaceExternalModels(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();

    StencilIRGenerator generator{ context };

    auto ir = generator.Generate(node);
    auto module = mlir::dyn_cast<mlir::ModuleOp>(ir.op);

    ScopedDiagnosticCollector diagnostics{ context };
    mlir::LogicalResult verificationResult = mlir::verify(module);
    if (failed(verificationResult)) {
        module->dump();
        throw CompilationError(diagnostics.TakeDiagnostics(), module);
    }
    return module;
}