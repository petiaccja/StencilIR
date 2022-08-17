#include "StencilOps.hpp"

#include <llvm/ADT/ArrayRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/ViewLikeInterface.h>
#include <mlir/Support/LLVM.h>

// clang-format: off
#define GET_OP_CLASSES
#include <StencilDialect/Stencil.cpp.inc>
// clang-format: on


using namespace mlir;
using namespace stencil;

//------------------------------------------------------------------------------
// StencilOp
//------------------------------------------------------------------------------

ParseResult StencilOp::parse(OpAsmParser& parser, OperationState& result) {
    auto buildFuncType =
        [](Builder& builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
           function_interface_impl::VariadicFlag,
           std::string&) { return builder.getFunctionType(argTypes, results); };

    return function_interface_impl::parseFunctionOp(
        parser, result, /*allowVariadic=*/false, buildFuncType);
}

void StencilOp::print(OpAsmPrinter& p) {
    function_interface_impl::printFunctionOp(p, *this, /*isVariadic=*/false);
}


//------------------------------------------------------------------------------
// ApplyOp
//------------------------------------------------------------------------------

LogicalResult ApplyOp::verifySymbolUses(SymbolTableCollection& symbolTable) {
    // Check that the callee attribute was specified.
    auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
    if (!fnAttr) {
        return emitOpError("requires a 'callee' symbol reference attribute");
    }
    StencilOp fn = symbolTable.lookupNearestSymbolFrom<StencilOp>(*this, fnAttr);
    if (!fn) {
        return emitOpError() << "'" << fnAttr.getValue()
                             << "' does not reference a valid function";
    }

    // Verify that outputs have same types as result types.
    if (getResultTypes().size() != getOutputs().size()) {
        return emitOpError("number of result types must equal number of output operands");
    }

    // Verify that the operand and result types match the callee.
    auto fnType = fn.getFunctionType();
    const size_t numCalleeParams = fnType.getNumInputs();
    const size_t numCalleeResults = fnType.getNumResults();
    const size_t numArgs = getInputs().size();
    const size_t numTargets = getOutputs().size();
    if (numCalleeParams != numArgs) {
        return emitOpError("number of arguments must match number of operands of callee");
    }
    if (numCalleeResults != numTargets) {
        return emitOpError("number of targets must match number of callee returns");
    }

    for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i) {
        if (getInputs()[i].getType() != fnType.getInput(i)) {
            return emitOpError("operand type mismatch: expected operand type ")
                   << fnType.getInput(i) << ", but provided "
                   << getOutputs()[i].getType() << " for operand number " << i;
        }
    }

    if (fnType.getNumResults() != getOutputs().size()) {
        return emitOpError("incorrect number of results for callee");
    }

    for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i) {
        mlir::Type calleeResultType = fnType.getResult(i);
        mlir::ShapedType targetType = getOutputs()[i].getType().dyn_cast<mlir::ShapedType>();
        if (targetType.getElementType() != calleeResultType) {
            auto diag = emitOpError("result type mismatch at index ") << i;
            diag.attachNote() << "      op result types: " << getResultTypes();
            diag.attachNote() << "function result types: " << fnType.getResults();
            return diag;
        }
    }

    return success();
}


void ApplyOp::build(::mlir::OpBuilder& odsBuilder,
                    ::mlir::OperationState& odsState,
                    ::llvm::StringRef callee,
                    ::mlir::ValueRange inputs,
                    ::mlir::ValueRange outputs) {
    return build(odsBuilder,
                 odsState,
                 outputs.getTypes(),
                 callee,
                 inputs,
                 outputs,
                 ::mlir::ValueRange{},
                 odsBuilder.getI64ArrayAttr({}));
}


void ApplyOp::build(::mlir::OpBuilder& odsBuilder,
                    ::mlir::OperationState& odsState,
                    ::llvm::StringRef callee,
                    ::mlir::ValueRange inputs,
                    ::mlir::ValueRange outputs,
                    ::llvm::ArrayRef<int64_t> static_offsets) {
    return build(odsBuilder,
                 odsState,
                 outputs.getTypes(),
                 callee,
                 inputs,
                 outputs,
                 ::mlir::ValueRange{},
                 odsBuilder.getI64ArrayAttr(static_offsets));
}


void ApplyOp::build(::mlir::OpBuilder& odsBuilder,
                    ::mlir::OperationState& odsState,
                    ::llvm::StringRef callee,
                    ::mlir::ValueRange inputs,
                    ::mlir::ValueRange outputs,
                    ::mlir::ValueRange offsets) {
    return build(odsBuilder,
                 odsState,
                 outputs.getTypes(),
                 callee,
                 inputs,
                 outputs,
                 offsets,
                 odsBuilder.getI64ArrayAttr({}));
}

::mlir::LogicalResult ApplyOp::verify() {
    const auto& outputTypes = getOutputs().getTypes();
    const auto& resultTypes = getResultTypes();

    // Same number of output operands and results
    if (outputTypes.size() != resultTypes.size()) {
        return emitOpError("must have equal number of output operands as results");
    }

    // Same types of output operands and results
    for (size_t i = 0; i < outputTypes.size(); ++i) {
        if (outputTypes[i].getTypeID() != resultTypes[i].getTypeID()) {
            return emitOpError("output operand must have the same types as results");
        }
    }

    // Either static or dynamic offsets
    if (!getOffsets().empty() && !getStaticOffsets().empty()) {
        return emitOpError("cannot have both static and dynamic offsets");
    }

    return success();
}

//------------------------------------------------------------------------------
// InvokeStencilOp
//------------------------------------------------------------------------------

LogicalResult InvokeStencilOp::verifySymbolUses(SymbolTableCollection& symbolTable) {
    // Check that the callee attribute was specified.
    auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
    if (!fnAttr)
        return emitOpError("requires a 'callee' symbol reference attribute");
    StencilOp fn = symbolTable.lookupNearestSymbolFrom<StencilOp>(*this, fnAttr);
    if (!fn)
        return emitOpError() << "'" << fnAttr.getValue()
                             << "' does not reference a valid function";

    // Verify that the operand and result types match the callee.
    auto fnType = fn.getFunctionType();
    if (fnType.getNumInputs() != getArguments().size())
        return emitOpError("incorrect number of operands for callee");

    for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
        if (getArguments()[i].getType() != fnType.getInput(i))
            return emitOpError("operand type mismatch: expected operand type ")
                   << fnType.getInput(i) << ", but provided "
                   << getArguments()[i].getType() << " for operand number " << i;

    if (fnType.getNumResults() != getNumResults())
        return emitOpError("incorrect number of results for callee");

    for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
        if (getResult(i).getType() != fnType.getResult(i)) {
            auto diag = emitOpError("result type mismatch at index ") << i;
            diag.attachNote() << "      op result types: " << getResultTypes();
            diag.attachNote() << "function result types: " << fnType.getResults();
            return diag;
        }

    return success();
}

FunctionType InvokeStencilOp::getCalleeType() {
    std::vector<Type> argumentTypes;
    for (const auto& arg : getArguments()) {
        argumentTypes.push_back(arg.getType());
    }
    return FunctionType::get(getContext(), argumentTypes, getResultTypes());
}