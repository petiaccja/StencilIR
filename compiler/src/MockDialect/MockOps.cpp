#include "MockOps.hpp"

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>

// clang-format: off
#define GET_OP_CLASSES
#include <MockDialect/Mock.cpp.inc>
// clang-format: on


using namespace mlir;
using namespace mock;

//------------------------------------------------------------------------------
// KernelCallOp
//------------------------------------------------------------------------------

LogicalResult KernelCallOp::verifySymbolUses(SymbolTableCollection& symbolTable) {
    // Check that the callee attribute was specified.
    auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
    if (!fnAttr)
        return emitOpError("requires a 'callee' symbol reference attribute");
    KernelFuncOp fn = symbolTable.lookupNearestSymbolFrom<KernelFuncOp>(*this, fnAttr);
    if (!fn)
        return emitOpError() << "'" << fnAttr.getValue()
                             << "' does not reference a valid function";

    // Verify that the operand and result types match the callee.
    auto fnType = fn.getFunctionType();
    if (fnType.getNumInputs() != getArguments().size()) {
        return emitOpError("number of arguments must match number of operands of callee");
    }
    if (fnType.getNumResults() != getTargets().size()) {
        return emitOpError("number of targets must match number of callee returns");
    }

    for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
        if (getOperand(i).getType() != fnType.getInput(i))
            return emitOpError("operand type mismatch: expected operand type ")
                   << fnType.getInput(i) << ", but provided "
                   << getOperand(i).getType() << " for operand number " << i;

    if (fnType.getNumResults() != getTargets().size())
        return emitOpError("incorrect number of results for callee");

    for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
        if (getTargets()[i].getType() != fnType.getResult(i)) {
            auto diag = emitOpError("result type mismatch at index ") << i;
            diag.attachNote() << "      op result types: " << getTargetTypes();
            diag.attachNote() << "function result types: " << fnType.getResults();
            return diag;
        }

    return success();
}

FunctionType KernelCallOp::getCalleeType() {
    return FunctionType::get(getContext(), getOperandTypes(), getTargetTypes());
}

mlir::TypeRange KernelCallOp::getTargetTypes() {
    llvm::ArrayRef<mlir::Type> targetTypes;
    for (const auto& target : getTargets()) {
        const auto targetType = target.getType();
        const auto memrefType = targetType.dyn_cast<mlir::MemRefType>();
        const auto elementType = memrefType.getElementType();
        targetTypes.vec().push_back(elementType);
    }

    return { targetTypes };
}