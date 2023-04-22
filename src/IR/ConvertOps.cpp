#include "ConvertOps.hpp"

#include "ConvertUtils.hpp"
#include "Converter.hpp"
#include "Ops.hpp"

#include <Diagnostics/Exception.hpp>
#include <Diagnostics/Handlers.hpp>
#include <Dialect/Stencil/IR/StencilOps.hpp>
#include <Dialect/Stencil/Transforms/BufferizableOpInterfaceImpl.hpp>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/Verifier.h>
#include <mlir/InitAllDialects.h>


namespace sir {


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
    const auto loc = ConvertLocation(builder, op.GetLocation());

    auto converted = builder.create<mlir::ModuleOp>(loc);

    builder.setInsertionPointToEnd(converted.getBody());
    for (const auto& op : op.GetRegions().front().GetOperations()) {
        converter(op);
    }

    return converted;
}


mlir::Operation* ConvertFuncOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.GetLocation());
    const auto& attr = std::any_cast<const ops::FuncAttr&>(op.GetAttributes());

    auto functionType = ConvertType(builder, *attr.signature).dyn_cast<mlir::FunctionType>();

    auto converted = builder.create<mlir::func::FuncOp>(loc, attr.name, functionType);
    const auto entryBlock = converted.addEntryBlock();
    converter.MapEntryBlock(op.GetRegions().front(), *entryBlock);

    converted.setVisibility(attr.isPublic ? mlir::SymbolTable::Visibility::Public : mlir::SymbolTable::Visibility::Private);
    builder.setInsertionPointToEnd(entryBlock);
    for (const auto& op : op.GetRegions().front().GetOperations()) {
        converter(op);
    }

    return converted;
}


mlir::Operation* ConvertStencilOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.GetLocation());
    const auto& attr = std::any_cast<const ops::StencilAttr&>(op.GetAttributes());

    auto functionType = ConvertType(builder, *attr.signature).dyn_cast<mlir::FunctionType>();

    auto converted = builder.create<stencil::StencilOp>(loc, attr.name, functionType, builder.getI64IntegerAttr(attr.numDims));

    const auto entryBlock = converted.addEntryBlock();
    converter.MapEntryBlock(op.GetRegions().front(), *entryBlock);

    converted.setVisibility(attr.isPublic ? mlir::SymbolTable::Visibility::Public : mlir::SymbolTable::Visibility::Private);
    builder.setInsertionPointToEnd(entryBlock);
    for (const auto& op : op.GetRegions().front().GetOperations()) {
        converter(op);
    }

    return converted;
}


mlir::Operation* ConvertReturnOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.GetLocation());

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
    const auto loc = ConvertLocation(builder, op.GetLocation());
    const auto& attr = std::any_cast<ops::CallAttr>(op.GetAttributes());
    const auto callee = mlir::StringRef{ attr.name };
    mlir::SmallVector<mlir::Type> results;
    std::transform(attr.results.begin(), attr.results.end(), std::back_inserter(results), [&builder](const auto& type) {
        return ConvertType(builder, *type);
    });
    return { builder.create<mlir::func::CallOp>(loc, callee, results, operands) };
}


mlir::Operation* ConvertApplyOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.GetLocation());
    const auto& attr = std::any_cast<const ops::ApplyAttr&>(op.GetAttributes());

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
    const auto loc = ConvertLocation(builder, op.GetLocation());
    const auto& attr = std::any_cast<const TypePtr&>(op.GetAttributes());

    mlir::Value value = operands[0];
    mlir::Type type = ConvertType(builder, *attr);
    auto cast = Cast(value, type, builder, loc);
    if (!cast) {
        std::vector<mlir::Diagnostic> diags;
        auto& diag = diags.emplace_back(loc, mlir::DiagnosticSeverity::Error);
        diag << "cannot convert value from `" << value.getType() << "` to `" << type << "`";
        throw CompilationError(diags);
    }

    auto wrapper = builder.create<mlir::scf::ExecuteRegionOp>(loc, type);
    auto& block = wrapper->getRegion(0).emplaceBlock();
    builder.setInsertionPointToEnd(&block);
    builder.create<mlir::scf::YieldOp>(loc, mlir::ValueRange{ cast.value() });
    return wrapper;
}


mlir::Operation* ConvertConstantOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.GetLocation());
    const auto& attr = std::any_cast<const ops::ConstantAttr&>(op.GetAttributes());

    const auto type = ConvertType(builder, *attr.type);

    if (std::dynamic_pointer_cast<IntegerType>(attr.type)) {
        const int64_t value = std::any_cast<int64_t>(attr.value);
        auto signlessType = builder.getIntegerType(type.dyn_cast<mlir::IntegerType>().getWidth());
        return { builder.create<mlir::arith::ConstantIntOp>(loc, value, signlessType) };
    }
    else if (std::dynamic_pointer_cast<IndexType>(attr.type)) {
        const int64_t value = std::any_cast<int64_t>(attr.value);
        return { builder.create<mlir::arith::ConstantIndexOp>(loc, value) };
    }
    else if (auto floatType = std::dynamic_pointer_cast<FloatType>(attr.type)) {
        const double value = std::any_cast<double>(attr.value);
        const auto apfloat = floatType->size == 32 ? mlir::APFloat(float(value)) : mlir::APFloat(double(value));
        return { builder.create<mlir::arith::ConstantFloatOp>(loc, apfloat, type.dyn_cast<mlir::FloatType>()) };
    }
    throw ArgumentTypeError{ loc, FormatType(type), 0 };
}


mlir::Operation* ConvertArithmeticOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.GetLocation());
    const auto& attr = std::any_cast<const ops::eArithmeticFunction&>(op.GetAttributes());

    auto [lhs, rhs] = std::tuple{ operands[0], operands[1] };
    bool isFloat = lhs.getType().isa<mlir::FloatType>();
    bool isUnsigned = lhs.getType().isUnsignedInteger();

    switch (attr) {
        case ops::eArithmeticFunction::ADD:
            return isFloat ? builder.create<mlir::arith::AddFOp>(loc, lhs, rhs)
                           : builder.create<mlir::arith::AddIOp>(loc, lhs, rhs);
        case ops::eArithmeticFunction::SUB:
            return isFloat ? builder.create<mlir::arith::SubFOp>(loc, lhs, rhs)
                           : builder.create<mlir::arith::SubIOp>(loc, lhs, rhs);
        case ops::eArithmeticFunction::MUL:
            return isFloat ? builder.create<mlir::arith::MulFOp>(loc, lhs, rhs)
                           : builder.create<mlir::arith::MulIOp>(loc, lhs, rhs);
        case ops::eArithmeticFunction::DIV:
            return isFloat      ? builder.create<mlir::arith::DivFOp>(loc, lhs, rhs)
                   : isUnsigned ? builder.create<mlir::arith::DivUIOp>(loc, lhs, rhs)
                                : builder.create<mlir::arith::DivSIOp>(loc, lhs, rhs);
        case ops::eArithmeticFunction::MOD:
            return isFloat      ? builder.create<mlir::arith::RemFOp>(loc, lhs, rhs)
                   : isUnsigned ? builder.create<mlir::arith::RemUIOp>(loc, lhs, rhs)
                                : builder.create<mlir::arith::RemSIOp>(loc, lhs, rhs);
        case ops::eArithmeticFunction::BIT_AND:
            return builder.create<mlir::arith::AndIOp>(loc, lhs, rhs);
        case ops::eArithmeticFunction::BIT_OR:
            return builder.create<mlir::arith::OrIOp>(loc, lhs, rhs);
        case ops::eArithmeticFunction::BIT_XOR:
            return builder.create<mlir::arith::XOrIOp>(loc, lhs, rhs);
        case ops::eArithmeticFunction::BIT_SHL:
            return builder.create<mlir::arith::ShLIOp>(loc, lhs, rhs);
        case ops::eArithmeticFunction::BIT_SHR:
            return isUnsigned ? builder.create<mlir::arith::ShRUIOp>(loc, lhs, rhs)
                              : builder.create<mlir::arith::ShRSIOp>(loc, lhs, rhs);
    }
    throw NotImplementedError("switch does not cover operator");
}



mlir::Operation* ConvertComparisonOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.GetLocation());
    const auto& attr = std::any_cast<const ops::eComparisonFunction&>(op.GetAttributes());

    auto [lhs, rhs] = std::tuple{ operands[0], operands[1] };
    bool isFloat = lhs.getType().isa<mlir::FloatType>();
    bool isUnsigned = lhs.getType().isUnsignedInteger();
    switch (attr) {
        case ops::eComparisonFunction::EQ:
            return isFloat ? builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ, lhs, rhs)
                           : builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::eq, lhs, rhs);
        case ops::eComparisonFunction::NEQ:
            return isFloat ? builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::ONE, lhs, rhs)
                           : builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne, lhs, rhs);
        case ops::eComparisonFunction::GT:
            return isFloat      ? builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGT, lhs, rhs)
                   : isUnsigned ? builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ugt, lhs, rhs)
                                : builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sgt, lhs, rhs);
        case ops::eComparisonFunction::LT:
            return isFloat      ? builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLT, lhs, rhs)
                   : isUnsigned ? builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, lhs, rhs)
                                : builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, lhs, rhs);
        case ops::eComparisonFunction::GTE:
            return isFloat      ? builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OGE, lhs, rhs)
                   : isUnsigned ? builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::uge, lhs, rhs)
                                : builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sge, lhs, rhs);
        case ops::eComparisonFunction::LTE:
            return isFloat      ? builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OLE, lhs, rhs)
                   : isUnsigned ? builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ule, lhs, rhs)
                                : builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sle, lhs, rhs);
    }
    throw NotImplementedError("switch does not cover operator");
}

mlir::Operation* ConvertMinOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.GetLocation());

    auto [lhs, rhs] = std::tuple{ operands[0], operands[1] };
    bool isFloat = lhs.getType().isa<mlir::FloatType>();
    bool isUnsigned = lhs.getType().isUnsignedInteger();

    return isFloat      ? builder.create<mlir::arith::MinFOp>(loc, lhs, rhs)
           : isUnsigned ? builder.create<mlir::arith::MinUIOp>(loc, lhs, rhs)
                        : builder.create<mlir::arith::MinSIOp>(loc, lhs, rhs);
}

mlir::Operation* ConvertMaxOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.GetLocation());

    auto [lhs, rhs] = std::tuple{ operands[0], operands[1] };
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
    const auto loc = ConvertLocation(builder, op.GetLocation());

    const mlir::Value condition = operands[0];

    auto insertionBlock = builder.getInsertionBlock();
    auto insertionPoint = builder.getInsertionPoint();

    // Create a block to infer result types.
    auto currentRegion = builder.getInsertionBlock()->getParent();
    auto& block = *builder.createBlock(currentRegion, currentRegion->end());

    builder.setInsertionPointToEnd(&block);
    for (const auto& op : op.GetRegions()[0].GetOperations()) {
        converter(op);
    }

    mlir::SmallVector<mlir::Type, 4> resultTypes;
    if (!block.empty()) {
        const auto resultTypesView = block.getTerminator()->getOperandTypes();
        resultTypes = { resultTypesView.begin(), resultTypesView.end() };
    }

    builder.setInsertionPoint(insertionBlock, insertionPoint);

    // Create the actual IfOp with result types and both blocks.
    const bool hasElseBlock = !op.GetRegions()[1].GetOperations().empty();
    auto converted = builder.create<mlir::scf::IfOp>(loc, resultTypes, condition, hasElseBlock);

    auto& thenBlock = *converted.thenBlock();
    block.moveBefore(&thenBlock);
    thenBlock.erase();

    if (hasElseBlock) {
        auto& elseBlock = *converted.elseBlock();
        elseBlock.clear();
        builder.setInsertionPointToEnd(&elseBlock);
        for (const auto& op : op.GetRegions()[1].GetOperations()) {
            converter(op);
        }
    }

    // Return the actual IfOp, not the deduction which has been erased.
    return converted;
}


mlir::Operation* ConvertForOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.GetLocation());

    const mlir::Value start = operands[0];
    const mlir::Value end = operands[1];
    const mlir::Value step = operands[2];
    const auto initArgs = mlir::ValueRange{ operands.begin() + 3, operands.end() };

    auto converted = builder.create<mlir::scf::ForOp>(loc, start, end, step, initArgs);

    auto& body = *converted.getBody();
    body.clear();
    converter.MapEntryBlock(op.GetRegions().front(), body);

    builder.setInsertionPointToEnd(&body);
    for (const auto& op : op.GetRegions().front().GetOperations()) {
        converter(op);
    }

    return converted;
}


mlir::Operation* ConvertYieldOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.GetLocation());
    return builder.create<mlir::scf::YieldOp>(loc, operands);
}


//------------------------------------------------------------------------------
// Tensor
//------------------------------------------------------------------------------


mlir::Operation* ConvertDimOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.GetLocation());
    const mlir::Value tensor = operands[0];
    const mlir::Value index = operands[1];
    return builder.create<mlir::tensor::DimOp>(loc, tensor, index);
}


mlir::Operation* ConvertAllocTensorOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.GetLocation());
    const auto& attr = std::any_cast<TypePtr>(op.GetAttributes());

    const auto elementType = ConvertType(builder, *attr);
    const auto sizes = operands;
    std::vector<int64_t> shape(sizes.size(), mlir::ShapedType::kDynamic);
    const auto type = mlir::RankedTensorType::get(shape, elementType);
    return { builder.create<mlir::tensor::EmptyOp>(loc, type, sizes) };
}


mlir::Operation* ConvertExtractSliceOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.GetLocation());

    const mlir::Value source = operands[0];

    const auto numDims = (operands.size() - 1) / 3;
    const auto offsets = operands.slice(1, numDims);
    const auto sizes = operands.slice(1 + numDims, numDims);
    const auto strides = operands.slice(1 + 2 * numDims, numDims);

    return { builder.create<mlir::tensor::ExtractSliceOp>(loc, source, offsets, sizes, strides) };
}


mlir::Operation* ConvertInsertSliceOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.GetLocation());

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
    const auto loc = ConvertLocation(builder, op.GetLocation());

    auto currentBlock = builder.getBlock();
    auto currentOp = currentBlock->getParentOp();
    stencil::StencilOp currentStencil = mlir::dyn_cast<stencil::StencilOp>(currentOp)
                                            ? mlir::dyn_cast<stencil::StencilOp>(currentOp)
                                            : currentOp->getParentOfType<stencil::StencilOp>();

    const int64_t numDims = currentStencil ? currentStencil.getNumDimensions() : 1;
    const auto indexType = mlir::VectorType::get({ numDims }, builder.getIndexType());

    return builder.create<stencil::IndexOp>(loc, indexType);
}


mlir::Operation* ConvertProjectOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.GetLocation());
    const auto positions = std::any_cast<const std::vector<int64_t>&>(op.GetAttributes());

    const mlir::Value index = operands[0];
    return builder.create<stencil::ProjectOp>(loc, index, positions);
}


mlir::Operation* ConvertJumpOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.GetLocation());
    const auto offset = std::any_cast<const std::vector<int64_t>&>(op.GetAttributes());

    const mlir::Value index = operands[0];
    const auto offsetAttr = builder.getI64ArrayAttr(offset);
    return builder.create<stencil::JumpOp>(loc, index.getType(), index, offsetAttr);
}


mlir::Operation* ConvertExtendOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.GetLocation());
    const auto position = std::any_cast<int64_t>(op.GetAttributes());

    const mlir::Value index = operands[0];
    const mlir::Value value = operands[1];
    return builder.create<stencil::ExtendOp>(loc, index, position, value);
}


mlir::Operation* ConvertExchangeOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.GetLocation());
    const auto position = std::any_cast<int64_t>(op.GetAttributes());

    const mlir::Value index = operands[0];
    const mlir::Value value = operands[1];
    return builder.create<stencil::ExchangeOp>(loc, index, position, value);
}


mlir::Operation* ConvertExtractOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.GetLocation());
    const auto position = std::any_cast<int64_t>(op.GetAttributes());

    const mlir::Value index = operands[0];
    return builder.create<stencil::ExtractOp>(loc, index, position);
}


mlir::Operation* ConvertSampleOp(Converter& converter, Operation op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.GetLocation());

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
    context.getOrLoadDialect<stencil::StencilDialect>();
    mlir::DialectRegistry registry;
    mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
    stencil::registerBufferizableOpInterfaceExternalModels(registry);
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();

    Converter converter{ context };

    // Module ops
    converter.RegisterOp<ops::ModuleOp>(&ConvertModuleOp);
    converter.RegisterOp<ops::FuncOp>(&ConvertFuncOp);
    converter.RegisterOp<ops::StencilOp>(&ConvertStencilOp);
    converter.RegisterOp<ops::ReturnOp>(&ConvertReturnOp);
    converter.RegisterOp<ops::CallOp>(&ConvertCallOp);
    converter.RegisterOp<ops::ApplyOp>(&ConvertApplyOp);

    // ALU ops
    converter.RegisterOp<ops::CastOp>(&ConvertCastOp);
    converter.RegisterOp<ops::ConstantOp>(&ConvertConstantOp);
    converter.RegisterOp<ops::ArithmeticOp>(&ConvertArithmeticOp);
    converter.RegisterOp<ops::ComparisonOp>(&ConvertComparisonOp);
    converter.RegisterOp<ops::MinOp>(&ConvertMinOp);
    converter.RegisterOp<ops::MaxOp>(&ConvertMaxOp);

    // Control flow ops
    converter.RegisterOp<ops::IfOp>(&ConvertIfOp);
    converter.RegisterOp<ops::ForOp>(&ConvertForOp);
    converter.RegisterOp<ops::YieldOp>(&ConvertYieldOp);

    // Tensor ops
    converter.RegisterOp<ops::DimOp>(&ConvertDimOp);
    converter.RegisterOp<ops::AllocTensorOp>(&ConvertAllocTensorOp);
    converter.RegisterOp<ops::ExtractSliceOp>(&ConvertExtractSliceOp);
    converter.RegisterOp<ops::InsertSliceOp>(&ConvertInsertSliceOp);

    // Index ops
    converter.RegisterOp<ops::IndexOp>(&ConvertIndexOp);
    converter.RegisterOp<ops::JumpOp>(&ConvertJumpOp);
    converter.RegisterOp<ops::ProjectOp>(&ConvertProjectOp);
    converter.RegisterOp<ops::ExtractOp>(&ConvertExtractOp);
    converter.RegisterOp<ops::ExtendOp>(&ConvertExtendOp);
    converter.RegisterOp<ops::ExchangeOp>(&ConvertExchangeOp);
    converter.RegisterOp<ops::SampleOp>(&ConvertSampleOp);


    auto converted = converter(op);

    ScopedDiagnosticCollector diagnostics{ context };
    mlir::LogicalResult verificationResult = mlir::verify(converted);
    if (failed(verificationResult)) {
        converted->dump();
        throw CompilationError(diagnostics.TakeDiagnostics(), mlir::dyn_cast<mlir::ModuleOp>(converted));
    }
    return converted;
}


} // namespace sir