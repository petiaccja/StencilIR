#include "StencilOps.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>

// clang-format: off
#define GET_OP_CLASSES
#include <StencilDialect/Stencil.cpp.inc>
// clang-format: on


using namespace mlir;
using namespace stencil;

//------------------------------------------------------------------------------
// KernelOp
//------------------------------------------------------------------------------

ParseResult KernelOp::parse(OpAsmParser& parser, OperationState& result) {
    auto buildFuncType =
        [](Builder& builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
           function_interface_impl::VariadicFlag,
           std::string&) { return builder.getFunctionType(argTypes, results); };

    return function_interface_impl::parseFunctionOp(
        parser, result, /*allowVariadic=*/false, buildFuncType);
}

void KernelOp::print(OpAsmPrinter& p) {
    function_interface_impl::printFunctionOp(p, *this, /*isVariadic=*/false);
}


//------------------------------------------------------------------------------
// LaunchKernelOp
//------------------------------------------------------------------------------

LogicalResult LaunchKernelOp::verifySymbolUses(SymbolTableCollection& symbolTable) {
    // Check that the callee attribute was specified.
    auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
    if (!fnAttr) {
        return emitOpError("requires a 'callee' symbol reference attribute");
    }
    KernelOp fn = symbolTable.lookupNearestSymbolFrom<KernelOp>(*this, fnAttr);
    if (!fn) {
        return emitOpError() << "'" << fnAttr.getValue()
                             << "' does not reference a valid function";
    }

    // Verify that the operand and result types match the callee.

    auto fnType = fn.getFunctionType();
    const size_t numCalleeParams = fnType.getNumInputs();
    const size_t numCalleeResults = fnType.getNumResults();
    const size_t numArgs = getArguments().size();
    const size_t numTargets = getTargets().size();
    if (numCalleeParams != numArgs) {
        return emitOpError("number of arguments must match number of operands of callee");
    }
    if (numCalleeResults != numTargets) {
        return emitOpError("number of targets must match number of callee returns");
    }

    for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i) {
        if (getArguments()[i].getType() != fnType.getInput(i)) {
            return emitOpError("operand type mismatch: expected operand type ")
                   << fnType.getInput(i) << ", but provided "
                   << getArguments()[i].getType() << " for operand number " << i;
        }
    }

    if (fnType.getNumResults() != getTargets().size()) {
        return emitOpError("incorrect number of results for callee");
    }

    for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i) {
        mlir::Type calleeResultType = fnType.getResult(i);
        mlir::MemRefType targetType = getTargets()[i].getType().dyn_cast<mlir::MemRefType>();
        if (targetType.getElementType() != calleeResultType) {
            auto diag = emitOpError("result type mismatch at index ") << i;
            diag.attachNote() << "      op result types: " << getTargetTypes();
            diag.attachNote() << "function result types: " << fnType.getResults();
            return diag;
        }
    }

    return success();
}

FunctionType LaunchKernelOp::getCalleeType() {
    return FunctionType::get(getContext(), getOperandTypes(), getTargetTypes());
}

mlir::TypeRange LaunchKernelOp::getTargetTypes() {
    llvm::ArrayRef<mlir::Type> targetTypes;
    for (const auto& target : getTargets()) {
        const auto targetType = target.getType();
        const auto memrefType = targetType.dyn_cast<mlir::MemRefType>();
        const auto elementType = memrefType.getElementType();
        targetTypes.vec().push_back(elementType);
    }

    return { targetTypes };
}