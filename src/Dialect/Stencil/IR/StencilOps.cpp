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
        if (inputType != calleeInputType && !(inputType.isa<ShapedType>() && calleeInputType.isa<ShapedType>())) {
            return emitOpError("operand type mismatch: expected operand type ")
                   << inputType << ", but provided "
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


//------------------------------------------------------------------------------
// ForeachElementOp
//------------------------------------------------------------------------------

void ForeachElementOp::build(OpBuilder& builder,
                         OperationState& result,
                         Value field,
                         IntegerAttr dimIndex,
                         ValueRange iterArgs,
                         BodyBuilderFn bodyBuilder) {
    result.addOperands({ field });
    result.addAttribute("dim", dimIndex);
    result.addOperands(iterArgs);
    for (Value v : iterArgs)
        result.addTypes(v.getType());
    Region* bodyRegion = result.addRegion();
    bodyRegion->push_back(new Block);
    Block& bodyBlock = bodyRegion->front();
    bodyBlock.addArgument(builder.getIndexType(), result.location);
    for (Value v : iterArgs)
        bodyBlock.addArgument(v.getType(), v.getLoc());

    // Create the default terminator if the builder is not provided and if the
    // iteration arguments are not provided. Otherwise, leave this to the caller
    // because we don't know which values to return from the loop.
    if (iterArgs.empty() && !bodyBuilder) {
        ForeachElementOp::ensureTerminator(*bodyRegion, builder, result.location);
    }
    else if (bodyBuilder) {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&bodyBlock);
        bodyBuilder(builder, result.location, bodyBlock.getArgument(0),
                    bodyBlock.getArguments().drop_front());
    }
}


LogicalResult ForeachElementOp::verify() {
    const int64_t dimIndex = getDim().getSExtValue();
    const int64_t fieldRank = getField().getType().dyn_cast<ShapedType>().getRank();
    if (!(dimIndex < fieldRank)) {
        return emitOpError("dim must be less than the field's rank");
    }

    const auto opNumResults = getNumResults();
    const auto numIterOperands = getNumIterOperands();
    if (opNumResults > 0 && numIterOperands != opNumResults) {
        return emitOpError("mismatch in number of loop-carried values and defined values");
    }
    return success();
}


LogicalResult ForeachElementOp::verifyRegions() {
    // Check that the body defines as single block argument for the induction
    // variable.
    auto* body = getBody();
    if (!body->getArgument(0).getType().isIndex()) {
        return emitOpError(
            "expected body first argument to be an index argument for "
            "the induction variable");
    }

    auto opNumResults = getNumResults();
    if (opNumResults == 0) {
        return success();
    }

    if (getNumRegionIterArgs() != opNumResults) {
        return emitOpError(
            "mismatch in number of basic block args and defined values");
    }

    auto iterOperands = getIterOperands();
    auto iterArgs = getRegionIterArgs();
    auto opResults = getResults();
    unsigned i = 0;
    for (auto e : llvm::zip(iterOperands, iterArgs, opResults)) {
        if (std::get<0>(e).getType() != std::get<2>(e).getType()) {
            return emitOpError() << "types mismatch between " << i
                                 << "th iter operand and defined value";
        }
        if (std::get<1>(e).getType() != std::get<2>(e).getType()) {
            return emitOpError() << "types mismatch between " << i
                                 << "th iter region arg and defined value";
        }
        i++;
    }
    return success();
}

Optional<Value> ForeachElementOp::getSingleInductionVar() { return getInductionVar(); }


Region& ForeachElementOp::getLoopBody() { return getRegion(); }

ForeachElementOp getForInductionVarOwner(Value val) {
    auto ivArg = val.dyn_cast<BlockArgument>();
    if (!ivArg)
        return ForeachElementOp();
    assert(ivArg.getOwner() && "unlinked block argument");
    auto* containingOp = ivArg.getOwner()->getParentOp();
    return dyn_cast_or_null<ForeachElementOp>(containingOp);
}

/// Return operands used when entering the region at 'index'. These operands
/// correspond to the loop iterator operands, i.e., those excluding the
/// induction variable. LoopOp only has one region, so 0 is the only valid value
/// for `index`.
OperandRange ForeachElementOp::getSuccessorEntryOperands(Optional<unsigned> index) {
    assert(index && *index == 0 && "invalid region index");

    // The initial operands map to the loop arguments after the induction
    // variable.
    return getInitArgs();
}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void ForeachElementOp::getSuccessorRegions(Optional<unsigned> index,
                                       ArrayRef<Attribute> operands,
                                       SmallVectorImpl<RegionSuccessor>& regions) {
    // If the predecessor is the ForOp, branch into the body using the iterator
    // arguments.
    if (!index) {
        regions.push_back(RegionSuccessor(&getLoopBody(), getRegionIterArgs()));
        return;
    }

    // Otherwise, the loop may branch back to itself or the parent operation.
    assert(*index == 0 && "expected loop region");
    regions.push_back(RegionSuccessor(&getLoopBody(), getRegionIterArgs()));
    regions.push_back(RegionSuccessor(getResults()));
}

} // namespace stencil