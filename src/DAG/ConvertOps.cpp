#include "ConvertOps.hpp"

#include "Converter.hpp"
#include "Ops.hpp"

#include <AST/Utility.hpp>
#include <Diagnostics/Exception.hpp>
#include <Diagnostics/Handlers.hpp>
#include <Dialect/Stencil/IR/StencilOps.hpp>
#include <Dialect/Stencil/Transforms/BufferizableOpInterfaceImpl.hpp>

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Verifier.h>
#include <mlir/InitAllDialects.h>


namespace dag {


static mlir::Location ConvertLocation(mlir::OpBuilder& builder, const std::optional<Location>& location) {
    if (location) {
        auto fileattr = builder.getStringAttr(location->file);
        return mlir::FileLineColLoc::get(fileattr, location->line, location->col);
    }
    return builder.getUnknownLoc();
}


//------------------------------------------------------------------------------
// Module, functions, stencils
//------------------------------------------------------------------------------

mlir::Operation* ConvertModuleOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.Location());

    auto converted = builder.create<mlir::ModuleOp>(loc);

    builder.setInsertionPointToEnd(converted.getBody());
    for (const auto& op : op.Regions().front().operations) {
        converter(op);
    }

    return converted;
}


mlir::Operation* ConvertFuncOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.Location());
    const auto& attr = std::any_cast<const FuncAttr&>(op.Attributes());

    auto functionType = ConvertType(builder, *attr.signature).dyn_cast<mlir::FunctionType>();

    auto converted = builder.create<mlir::func::FuncOp>(loc, attr.name, functionType);
    const auto entryBlock = converted.addEntryBlock();
    converter.MapEntryBlock(op.Regions().front(), *entryBlock);

    converted.setVisibility(attr.isPublic ? mlir::SymbolTable::Visibility::Public : mlir::SymbolTable::Visibility::Private);
    builder.setInsertionPointToEnd(entryBlock);
    for (const auto& op : op.Regions().front().operations) {
        converter(op);
    }

    return converted;
}


mlir::Operation* ConvertStencilOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.Location());
    const auto& attr = std::any_cast<const StencilAttr&>(op.Attributes());

    auto functionType = ConvertType(builder, *attr.signature).dyn_cast<mlir::FunctionType>();

    auto converted = builder.create<stencil::StencilOp>(loc, attr.name, functionType, mlir::APInt(64, attr.numDims));

    const auto entryBlock = converted.addEntryBlock();
    converter.MapEntryBlock(op.Regions().front(), *entryBlock);

    converted.setVisibility(attr.isPublic ? mlir::SymbolTable::Visibility::Public : mlir::SymbolTable::Visibility::Private);
    builder.setInsertionPointToEnd(entryBlock);
    for (const auto& op : op.Regions().front().operations) {
        converter(op);
    }

    return converted;
}


mlir::Operation* ConvertReturnOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.Location());

    auto parentOp = builder.getInsertionBlock()->getParentOp();
    assert(parentOp);
    const bool isFuncOp = mlir::isa<mlir::func::FuncOp>(parentOp)
                          || parentOp->getParentOfType<mlir::func::FuncOp>();

    if (isFuncOp) {
        return builder.create<mlir::func::ReturnOp>(loc, operands);
    }
    else {
        return builder.create<stencil::ReturnOp>(loc, operands);
    }
}


mlir::Operation* ConvertCallOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.Location());
    const auto& attr = std::any_cast<CallAttr>(op.Attributes());

    const auto parentOp = builder.getBlock()->getParentOp();
    const auto calleeAttr = builder.getStringAttr(attr.name);
    const auto calleeOp = mlir::SymbolTable::lookupNearestSymbolFrom(parentOp, calleeAttr);
    if (!calleeOp) {
        throw UndefinedSymbolError{ loc, attr.name };
    }
    const auto calleeFuncOp = mlir::dyn_cast<mlir::func::FuncOp>(calleeOp);
    if (!calleeFuncOp) {
        throw UndefinedSymbolError{ loc, attr.name };
    }
    return { builder.create<mlir::func::CallOp>(loc, calleeFuncOp, operands) };
}


mlir::Operation* ConvertApplyOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.Location());
    const auto& attr = std::any_cast<const ApplyAttr&>(op.Attributes());

    const auto callee = mlir::StringRef(attr.name);
    auto inputs = operands.slice(0, attr.numInputs);
    auto outputs = operands.slice(attr.numInputs, attr.numOutputs);
    auto offsets = operands.slice(attr.numInputs + attr.numOutputs, attr.numOffsets);

    auto apply = builder.create<stencil::ApplyOp>(loc,
                                                  callee,
                                                  inputs,
                                                  outputs,
                                                  offsets,
                                                  attr.staticOffsets);

    return apply;
}


//------------------------------------------------------------------------------
// Arithmetic-logic
//------------------------------------------------------------------------------


mlir::Operation* ConvertCastOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.Location());
    const auto& attr = std::any_cast<const ast::TypePtr&>(op.Attributes());

    mlir::Value expr = operands[0];
    mlir::Type type = ConvertType(builder, *attr);

    auto makeEmptyOp = [&]() {
        auto noop = builder.create<mlir::scf::ExecuteRegionOp>(loc, mlir::TypeRange{ type });
        auto& block = noop.getRegion().emplaceBlock();
        builder.setInsertionPointToEnd(&block);
        builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{ expr });
        return noop;
    };

    if (expr.getType() == type) {
        return makeEmptyOp();
    }

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
            return makeEmptyOp();
        }
    }
    if (itype && iexpr) {
        bool isSigned = !(itype.isUnsigned() && iexpr.isUnsigned());
        if (ftype.getWidth() > fexpr.getWidth()) {
            return isSigned ? builder.create<mlir::arith::ExtSIOp>(loc, type, expr)
                            : builder.create<mlir::arith::ExtUIOp>(loc, type, expr);
        }
        else if (ftype.getWidth() < fexpr.getWidth()) {
            return { builder.create<mlir::arith::TruncIOp>(loc, type, expr) };
        }
        else {
            return makeEmptyOp();
        }
    }
    if (itype && fexpr) {
        return itype.isUnsigned() ? builder.create<mlir::arith::FPToUIOp>(loc, type, expr)
                                  : builder.create<mlir::arith::FPToSIOp>(loc, type, expr);
    }
    if (ftype && iexpr) {
        return iexpr.isUnsigned() ? builder.create<mlir::arith::UIToFPOp>(loc, type, expr)
                                  : builder.create<mlir::arith::SIToFPOp>(loc, type, expr);
    }
    throw std::invalid_argument("No conversion implemented between given types.");
}


mlir::Operation* ConvertConstantOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.Location());
    const auto& attr = std::any_cast<const ConstantAttr&>(op.Attributes());

    const auto type = ConvertType(builder, *attr.type);

    if (std::dynamic_pointer_cast<ast::IntegerType>(attr.type)) {
        const int64_t value = std::any_cast<int64_t>(attr.value);
        auto signlessType = builder.getIntegerType(type.dyn_cast<mlir::IntegerType>().getWidth());
        return { builder.create<mlir::arith::ConstantIntOp>(loc, value, signlessType) };
    }
    else if (std::dynamic_pointer_cast<ast::IndexType>(attr.type)) {
        const int64_t value = std::any_cast<int64_t>(attr.value);
        return { builder.create<mlir::arith::ConstantIndexOp>(loc, value) };
    }
    else if (auto floatType = std::dynamic_pointer_cast<ast::FloatType>(attr.type)) {
        const double value = std::any_cast<double>(attr.value);
        const auto apfloat = floatType->size == 32 ? mlir::APFloat(float(value)) : mlir::APFloat(double(value));
        return { builder.create<mlir::arith::ConstantFloatOp>(loc, apfloat, type.dyn_cast<mlir::FloatType>()) };
    }
    throw ArgumentTypeError{ loc, FormatType(type), 0 };
}


mlir::Operation* ConvertArithmeticOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.Location());
    const auto& attr = std::any_cast<const eArithmeticFunction&>(op.Attributes());

    auto [lhs, rhs] = PromoteToCommonType(builder, loc, operands[0], operands[1]);
    bool isFloat = lhs.getType().isa<mlir::FloatType>();
    bool isUnsigned = lhs.getType().isUnsignedInteger();

    switch (attr) {
        case eArithmeticFunction::ADD:
            return isFloat ? builder.create<mlir::arith::AddFOp>(loc, lhs, rhs)
                           : builder.create<mlir::arith::AddIOp>(loc, lhs, rhs);
        case eArithmeticFunction::SUB:
            return isFloat ? builder.create<mlir::arith::SubFOp>(loc, lhs, rhs)
                           : builder.create<mlir::arith::SubIOp>(loc, lhs, rhs);
        case eArithmeticFunction::MUL:
            return isFloat ? builder.create<mlir::arith::MulFOp>(loc, lhs, rhs)
                           : builder.create<mlir::arith::MulIOp>(loc, lhs, rhs);
        case eArithmeticFunction::DIV:
            return isFloat      ? builder.create<mlir::arith::DivFOp>(loc, lhs, rhs)
                   : isUnsigned ? builder.create<mlir::arith::DivUIOp>(loc, lhs, rhs)
                                : builder.create<mlir::arith::DivSIOp>(loc, lhs, rhs);
        case eArithmeticFunction::MOD:
            return isFloat      ? builder.create<mlir::arith::RemFOp>(loc, lhs, rhs)
                   : isUnsigned ? builder.create<mlir::arith::RemUIOp>(loc, lhs, rhs)
                                : builder.create<mlir::arith::RemSIOp>(loc, lhs, rhs);
        case eArithmeticFunction::BIT_AND:
            return builder.create<mlir::arith::AndIOp>(loc, lhs, rhs);
        case eArithmeticFunction::BIT_OR:
            return builder.create<mlir::arith::OrIOp>(loc, lhs, rhs);
        case eArithmeticFunction::BIT_XOR:
            return builder.create<mlir::arith::XOrIOp>(loc, lhs, rhs);
        case eArithmeticFunction::BIT_SHL:
            return builder.create<mlir::arith::ShLIOp>(loc, lhs, rhs);
        case eArithmeticFunction::BIT_SHR:
            return isUnsigned ? builder.create<mlir::arith::ShRUIOp>(loc, lhs, rhs)
                              : builder.create<mlir::arith::ShRSIOp>(loc, lhs, rhs);
    }
    throw NotImplementedError("switch does not cover operator");
}



mlir::Operation* ConvertComparisonOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.Location());
    const auto& attr = std::any_cast<const eComparisonFunction&>(op.Attributes());

    auto [lhs, rhs] = PromoteToCommonType(builder, loc, operands[0], operands[1]);
    bool isFloat = lhs.getType().isa<mlir::FloatType>();
    bool isUnsigned = lhs.getType().isUnsignedInteger();
    switch (attr) {
        case eComparisonFunction::EQ:
            return isFloat ? builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ, lhs, rhs)
                           : builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, lhs, rhs);
        case eComparisonFunction::NEQ:
            return isFloat ? builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::ONE, lhs, rhs)
                           : builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, lhs, rhs);
        case eComparisonFunction::GT:
            return isFloat      ? builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, lhs, rhs)
                   : isUnsigned ? builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ugt, lhs, rhs)
                                : builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sgt, lhs, rhs);
        case eComparisonFunction::LT:
            return isFloat      ? builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, lhs, rhs)
                   : isUnsigned ? builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, lhs, rhs)
                                : builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, lhs, rhs);
        case eComparisonFunction::GTE:
            return isFloat      ? builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGE, lhs, rhs)
                   : isUnsigned ? builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::uge, lhs, rhs)
                                : builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, lhs, rhs);
        case eComparisonFunction::LTE:
            return isFloat      ? builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLE, lhs, rhs)
                   : isUnsigned ? builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ule, lhs, rhs)
                                : builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sle, lhs, rhs);
    }
    throw NotImplementedError("switch does not cover operator");
}

mlir::Operation* ConvertMinOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.Location());

    auto [lhs, rhs] = PromoteToCommonType(builder, loc, operands[0], operands[1]);
    bool isFloat = lhs.getType().isa<mlir::FloatType>();
    bool isUnsigned = lhs.getType().isUnsignedInteger();

    return isFloat      ? builder.create<mlir::arith::MinFOp>(loc, lhs, rhs)
           : isUnsigned ? builder.create<mlir::arith::MinUIOp>(loc, lhs, rhs)
                        : builder.create<mlir::arith::MinSIOp>(loc, lhs, rhs);
}

mlir::Operation* ConvertMaxOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.Location());

    auto [lhs, rhs] = PromoteToCommonType(builder, loc, operands[0], operands[1]);
    bool isFloat = lhs.getType().isa<mlir::FloatType>();
    bool isUnsigned = lhs.getType().isUnsignedInteger();

    return isFloat      ? builder.create<mlir::arith::MaxFOp>(loc, lhs, rhs)
           : isUnsigned ? builder.create<mlir::arith::MaxUIOp>(loc, lhs, rhs)
                        : builder.create<mlir::arith::MaxSIOp>(loc, lhs, rhs);
}


//------------------------------------------------------------------------------
// Control flow
//------------------------------------------------------------------------------

mlir::Operation* ConvertIfOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.Location());

    const mlir::Value condition = operands[0];

    auto insertionBlock = builder.getInsertionBlock();
    auto insertionPoint = builder.getInsertionPoint();

    // Create a block to infer result types.
    auto currentRegion = builder.getInsertionBlock()->getParent();
    auto& block = *builder.createBlock(currentRegion, currentRegion->end());

    builder.setInsertionPointToEnd(&block);
    for (const auto& op : op.Regions()[0].operations) {
        converter(op);
    }

    mlir::SmallVector<mlir::Type, 4> resultTypes;
    if (!block.empty()) {
        const auto resultTypesView = block.getTerminator()->getOperandTypes();
        resultTypes = { resultTypesView.begin(), resultTypesView.end() };
    }

    builder.setInsertionPoint(insertionBlock, insertionPoint);

    // Create the actual IfOp with result types and both blocks.
    const bool hasElseBlock = !op.Regions()[1].operations.empty();
    auto converted = builder.create<mlir::scf::IfOp>(loc, resultTypes, condition, hasElseBlock);

    auto& thenBlock = *converted.thenBlock();
    block.moveBefore(&thenBlock);
    thenBlock.erase();

    if (hasElseBlock) {
        auto& elseBlock = *converted.elseBlock();
        elseBlock.clear();
        builder.setInsertionPointToEnd(&elseBlock);
        for (const auto& op : op.Regions()[1].operations) {
            converter(op);
        }
    }

    // Return the actual IfOp, not the deduction which has been erased.
    return converted;
}


mlir::Operation* ConvertForOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.Location());

    const mlir::Value start = operands[0];
    const mlir::Value end = operands[1];
    const mlir::Value step = operands[2];
    const auto initArgs = mlir::ValueRange{ operands.begin() + 3, operands.end() };

    auto converted = builder.create<mlir::scf::ForOp>(loc, start, end, step, initArgs);

    auto& body = *converted.getBody();
    body.clear();
    converter.MapEntryBlock(op.Regions().front(), body);

    builder.setInsertionPointToEnd(&body);
    for (const auto& op : op.Regions().front().operations) {
        converter(op);
    }

    return converted;
}


mlir::Operation* ConvertYieldOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.Location());
    return builder.create<mlir::scf::YieldOp>(loc, operands);
}


//------------------------------------------------------------------------------
// Tensor
//------------------------------------------------------------------------------


mlir::Operation* ConvertDimOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.Location());
    const mlir::Value tensor = operands[0];
    const mlir::Value index = operands[1];
    return builder.create<mlir::tensor::DimOp>(loc, tensor, index);
}


mlir::Operation* ConvertAllocTensorOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.Location());
    const auto& attr = std::any_cast<ast::TypePtr>(op.Attributes());

    const auto elementType = ConvertType(builder, *attr);
    const auto sizes = operands;
    std::vector<int64_t> shape(sizes.size(), mlir::ShapedType::kDynamicSize);
    const auto type = mlir::RankedTensorType::get(shape, elementType);
    return { builder.create<mlir::bufferization::AllocTensorOp>(loc, type, sizes) };
}


mlir::Operation* ConvertExtractSliceOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.Location());

    const mlir::Value source = operands[0];

    const auto numDims = (operands.size() - 1) / 3;
    const auto offsets = operands.slice(1, numDims);
    const auto sizes = operands.slice(1 + numDims, numDims);
    const auto strides = operands.slice(1 + 2 * numDims, numDims);

    return { builder.create<mlir::tensor::ExtractSliceOp>(loc, source, offsets, sizes, strides) };
}


mlir::Operation* ConvertInsertSliceOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.Location());

    const mlir::Value source = operands[0];
    const mlir::Value dest = operands[1];

    const auto numDims = (operands.size() - 2) / 3;
    const auto offsets = operands.slice(2, numDims);
    const auto sizes = operands.slice(2 + numDims, numDims);
    const auto strides = operands.slice(2 + 2 * numDims, numDims);

    return { builder.create<mlir::tensor::InsertSliceOp>(loc, source, dest, offsets, sizes, strides) };
}

//------------------------------------------------------------------------------
// Stencil
//------------------------------------------------------------------------------

mlir::Operation* ConvertIndexOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.Location());

    auto currentBlock = builder.getBlock();
    auto currentOp = currentBlock->getParentOp();
    stencil::StencilOp currentStencil = mlir::dyn_cast<stencil::StencilOp>(currentOp)
                                            ? mlir::dyn_cast<stencil::StencilOp>(currentOp)
                                            : currentOp->getParentOfType<stencil::StencilOp>();

    const int64_t numDims = currentStencil ? currentStencil.getNumDimensions().getSExtValue() : 1;
    const auto indexType = mlir::VectorType::get({ numDims }, builder.getIndexType());

    return builder.create<stencil::IndexOp>(loc, indexType);
}


mlir::Operation* ConvertProjectOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.Location());
    const auto positions = std::any_cast<const std::vector<int64_t>&>(op.Attributes());

    const mlir::Value index = operands[0];
    return builder.create<stencil::ProjectOp>(loc, index, positions);
}


mlir::Operation* ConvertJumpOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.Location());
    const auto offset = std::any_cast<const std::vector<int64_t>&>(op.Attributes());

    const mlir::Value index = operands[0];
    const auto offsetAttr = builder.getI64ArrayAttr(offset);
    return builder.create<stencil::JumpOp>(loc, index.getType(), index, offsetAttr);
}


mlir::Operation* ConvertExtendOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.Location());
    const auto position = std::any_cast<int64_t>(op.Attributes());

    const mlir::Value index = operands[0];
    const mlir::Value value = operands[1];
    return builder.create<stencil::ExtendOp>(loc, index, position, value);
}


mlir::Operation* ConvertExchangeOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.Location());
    const auto position = std::any_cast<int64_t>(op.Attributes());

    const mlir::Value index = operands[0];
    const mlir::Value value = operands[1];
    return builder.create<stencil::ExchangeOp>(loc, index, position, value);
}


mlir::Operation* ConvertExtractOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.Location());
    const auto position = std::any_cast<int64_t>(op.Attributes());

    const mlir::Value index = operands[0];
    return builder.create<stencil::ExtractOp>(loc, index, position);
}


mlir::Operation* ConvertSampleOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.Location());

    const mlir::Value field = operands[0];
    const mlir::Value index = operands[1];

    auto fieldType = field.getType().dyn_cast<mlir::ShapedType>();
    auto elementType = fieldType ? fieldType.getElementType() : builder.getNoneType();

    return builder.create<stencil::SampleOp>(loc, elementType, field, index);
}


//------------------------------------------------------------------------------
// Generic
//------------------------------------------------------------------------------

mlir::Operation* ConvertOperation(mlir::MLIRContext& context, Operation op) {
    mlir::registerAllDialects(context);
    mlir::DialectRegistry registry;
    registry.insert<stencil::StencilDialect>();
    mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
    stencil::registerBufferizableOpInterfaceExternalModels(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();


    Converter converter{ context };

    // Module ops
    converter.RegisterOp<ModuleOp>(&ConvertModuleOp);
    converter.RegisterOp<FuncOp>(&ConvertFuncOp);
    converter.RegisterOp<StencilOp>(&ConvertStencilOp);
    converter.RegisterOp<ReturnOp>(&ConvertReturnOp);
    converter.RegisterOp<CallOp>(&ConvertCallOp);
    converter.RegisterOp<ApplyOp>(&ConvertApplyOp);

    // ALU ops
    converter.RegisterOp<CastOp>(&ConvertCastOp);
    converter.RegisterOp<ConstantOp>(&ConvertConstantOp);
    converter.RegisterOp<ArithmeticOp>(&ConvertArithmeticOp);
    converter.RegisterOp<ComparisonOp>(&ConvertComparisonOp);
    converter.RegisterOp<MinOp>(&ConvertMinOp);
    converter.RegisterOp<MaxOp>(&ConvertMaxOp);

    // Control flow ops
    converter.RegisterOp<IfOp>(&ConvertIfOp);
    converter.RegisterOp<ForOp>(&ConvertForOp);
    converter.RegisterOp<YieldOp>(&ConvertYieldOp);

    // Tensor ops
    converter.RegisterOp<DimOp>(&ConvertDimOp);
    converter.RegisterOp<AllocTensorOp>(&ConvertAllocTensorOp);
    converter.RegisterOp<ExtractSliceOp>(&ConvertExtractSliceOp);
    converter.RegisterOp<InsertSliceOp>(&ConvertInsertSliceOp);

    // Index ops
    converter.RegisterOp<IndexOp>(&ConvertIndexOp);
    converter.RegisterOp<JumpOp>(&ConvertJumpOp);
    converter.RegisterOp<ProjectOp>(&ConvertProjectOp);
    converter.RegisterOp<ExtractOp>(&ConvertExtractOp);
    converter.RegisterOp<ExtendOp>(&ConvertExtendOp);
    converter.RegisterOp<ExchangeOp>(&ConvertExchangeOp);
    converter.RegisterOp<SampleOp>(&ConvertSampleOp);


    auto converted = converter(op);

    ScopedDiagnosticCollector diagnostics{ context };
    mlir::LogicalResult verificationResult = mlir::verify(converted);
    if (failed(verificationResult)) {
        converted->dump();
        throw CompilationError(diagnostics.TakeDiagnostics(), mlir::dyn_cast<mlir::ModuleOp>(converted));
    }
    return converted;
}


} // namespace dag