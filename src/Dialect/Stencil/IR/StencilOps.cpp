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
#include <Stencil/IR/StencilDialect.cpp.inc>
#define GET_OP_CLASSES
#include <Stencil/IR/Stencil.cpp.inc>
// clang-format: on


void stencil::StencilDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <Stencil/IR/Stencil.cpp.inc>
        >();
}

namespace stencil {
using namespace mlir;

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

    // Verify that the dimensions match up
    if (getNumResults() > 0) {
        const auto numStencilDims = fn.getNumDimensions().getSExtValue();
        const auto numMyDims = getResultTypes()[0].dyn_cast<mlir::ShapedType>().getRank();
        if (numStencilDims != numMyDims) {
            return emitOpError() << "number of stencil dimensions (" << numStencilDims << ")"
                                 << " does not equal number of output dimensions (" << numMyDims << ")";
        }
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
        mlir::Type inputType = getInputs()[i].getType();
        mlir::Type calleeInputType = fnType.getInput(i);
        if (inputType != calleeInputType) {
            return emitOpError("operand type mismatch: expected operand type ")
                   << inputType << ", but callee's operand type is "
                   << calleeInputType << " for operand number " << i;
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


static mlir::SmallVector<mlir::Type, 8> InferApplyOpResultTypes(mlir::ValueRange outputs) {
    if (!outputs.empty()) {
        auto types = outputs.getTypes();
        if (types.front().isa<mlir::TensorType>()) {
            return { types.begin(), types.end() };
        }
    }
    return {};
}


void ApplyOp::build(::mlir::OpBuilder& odsBuilder,
                    ::mlir::OperationState& odsState,
                    ::llvm::StringRef callee,
                    ::mlir::ValueRange inputs,
                    ::mlir::ValueRange outputs) {
    return build(odsBuilder,
                 odsState,
                 InferApplyOpResultTypes(outputs),
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
                 InferApplyOpResultTypes(outputs),
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
                 InferApplyOpResultTypes(outputs),
                 callee,
                 inputs,
                 outputs,
                 offsets,
                 odsBuilder.getI64ArrayAttr({}));
}

void ApplyOp::build(::mlir::OpBuilder& odsBuilder,
                    ::mlir::OperationState& odsState,
                    ::llvm::StringRef callee,
                    ::mlir::ValueRange inputs,
                    ::mlir::ValueRange outputs,
                    ::mlir::ValueRange offsets,
                    ::llvm::ArrayRef<int64_t> static_offsets) {
    return build(odsBuilder,
                 odsState,
                 InferApplyOpResultTypes(outputs),
                 callee,
                 inputs,
                 outputs,
                 offsets,
                 odsBuilder.getI64ArrayAttr(static_offsets));
}

::mlir::LogicalResult ApplyOp::verify() {
    const auto& outputTypes = getOutputs().getTypes();
    const auto& resultTypes = getResultTypes();

    // Either all output operands are tensors or memrefs
    bool isAllTensor = std::all_of(outputTypes.begin(), outputTypes.end(), [](mlir::Type type) {
        return type.isa<mlir::TensorType>();
    });
    bool isAllMemref = std::all_of(outputTypes.begin(), outputTypes.end(), [](mlir::Type type) {
        return type.isa<mlir::MemRefType>();
    });
    if (!isAllTensor && !isAllMemref) {
        return emitOpError("output operands must be either all tensors or all memrefs, not mixed");
    }

    if (isAllTensor) {
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
    }
    if (isAllMemref) {
        if (resultTypes.size() != 0) {
            return emitOpError("ops with memref semantics cannot have results");
        }
    }

    // Either static or dynamic offsets
    if (!getOffsets().empty() && !getStaticOffsets().empty()) {
        return emitOpError("cannot have both static and dynamic offsets");
    }

    return success();
}

void ApplyOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>& effects) {
    for (const auto& input : getInputs()) {
        if (input.getType().isa<ShapedType>()) {
            effects.emplace_back(MemoryEffects::Read::get(), input, SideEffects::DefaultResource::get());
        }
    }
    for (const auto& output : getOutputs()) {
        effects.emplace_back(MemoryEffects::Write::get(), output, SideEffects::DefaultResource::get());
    }
    auto stencilAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
    auto callee = SymbolTable::lookupNearestSymbolFrom(*this, stencilAttr.getAttr());
    if (auto stencil = mlir::dyn_cast<StencilOp>(callee)) {
        auto& body = stencil.getRegion();
        bool hasReadEffects = false;
        bool hasWriteEffects = false;
        body.walk([&](Operation* nestedOp) {
                if (auto effectInterface = mlir::dyn_cast<MemoryEffectOpInterface>(nestedOp)) {
                    hasReadEffects = hasReadEffects || effectInterface.hasEffect<MemoryEffects::Read>();
                    hasWriteEffects = hasWriteEffects || effectInterface.hasEffect<MemoryEffects::Write>();
                    hasWriteEffects = hasWriteEffects || nestedOp->hasTrait<::mlir::OpTrait::HasRecursiveSideEffects>();
                }
                else {
                    hasWriteEffects = true;
                }
                return !hasWriteEffects ? WalkResult::advance() : WalkResult::interrupt();
            })
            .wasInterrupted();
        if (hasReadEffects) {
            effects.emplace_back(MemoryEffects::Read::get());
        }
        if (hasWriteEffects) {
            effects.emplace_back(MemoryEffects::Write::get());
        }
    }
}


//------------------------------------------------------------------------------
// InvokeOp
//------------------------------------------------------------------------------

LogicalResult InvokeOp::verifySymbolUses(SymbolTableCollection& symbolTable) {
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

FunctionType InvokeOp::getCalleeType() {
    std::vector<Type> argumentTypes;
    for (const auto& arg : getArguments()) {
        argumentTypes.push_back(arg.getType());
    }
    return FunctionType::get(getContext(), argumentTypes, getResultTypes());
}


} // namespace stencil