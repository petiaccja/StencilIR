#include "BufferizableOpInterfaceImpl.hpp"

#include "../IR/StencilOps.hpp"

#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h>

#include <algorithm>
#include <iostream>


using namespace mlir;
using namespace bufferization;

namespace stencil {


FailureOr<std::vector<Value>> BufferizeValueRange(RewriterBase& rewriter,
                                                  ValueRange maybeTensors,
                                                  const BufferizationOptions& options) {
    std::vector<Value> maybeMemrefs;
    for (auto value : maybeTensors) {
        if (value.getType().isa<TensorType>()) {
            FailureOr<Value> memref = getBuffer(rewriter, value, options);
            if (failed(memref)) {
                return failure();
            }
            maybeMemrefs.push_back(*memref);
        }
        else {
            maybeMemrefs.push_back(value);
        }
    }
    return maybeMemrefs;
}


template <class TrivialOpT>
struct TrivialOpInterface
    : public BufferizableOpInterface::ExternalModel<TrivialOpInterface<TrivialOpT>,
                                                    TrivialOpT> {
    bool bufferizesToMemoryRead(Operation* op,
                                OpOperand& opOperand,
                                const AnalysisState& state) const {
        return true;
    }

    bool bufferizesToMemoryWrite(Operation* op,
                                 OpOperand& opOperand,
                                 const AnalysisState& state) const {
        return false;
    }

    SmallVector<OpResult> getAliasingOpResult(Operation* op,
                                              OpOperand& opOperand,
                                              const AnalysisState& state) const {
        return {};
    }

    BufferRelation bufferRelation(Operation* op,
                                  OpResult opResult,
                                  const AnalysisState& state) const {
        return BufferRelation::None;
    }

    LogicalResult bufferize(Operation* op,
                            RewriterBase& rewriter,
                            const BufferizationOptions& options) const {
        const auto& memrefOperandsOrFailure = BufferizeValueRange(rewriter, op->getOperands(), options);
        if (failed(memrefOperandsOrFailure)) {
            return memrefOperandsOrFailure;
        }
        const auto& memrefOperands = *memrefOperandsOrFailure;


        replaceOpWithNewBufferizedOp<TrivialOpT>(rewriter, op,
                                                 op->getResultTypes(),
                                                 memrefOperands,
                                                 op->getAttrs());

        return success();
    }
};


struct ApplyOpInterface : public BufferizableOpInterface::ExternalModel<ApplyOpInterface, ApplyOp> {
    bool bufferizesToMemoryRead(Operation* op,
                                OpOperand& opOperand,
                                const AnalysisState& state) const {
        auto applyOp = mlir::dyn_cast<ApplyOp>(op);

        const auto& inputs = applyOp.getInputs();
        return std::any_of(inputs.begin(), inputs.end(), [&opOperand](Value input) {
            return opOperand.is(input);
        });
    }

    bool bufferizesToMemoryWrite(Operation* op,
                                 OpOperand& opOperand,
                                 const AnalysisState& state) const {
        auto applyOp = mlir::dyn_cast<ApplyOp>(op);

        const auto& outputs = applyOp.getOutputs();
        return std::any_of(outputs.begin(), outputs.end(), [&opOperand](Value output) {
            return opOperand.is(output);
        });
    }

    SmallVector<OpOperand*> getAliasingOpOperand(Operation* op, OpResult opResult,
                                                 const AnalysisState& state) const {
        auto applyOp = mlir::dyn_cast<ApplyOp>(op);

        const auto operandIndex = applyOp.getInputs().size() + opResult.getResultNumber();
        return { &applyOp->getOpOperand(operandIndex) };
    }

    SmallVector<OpResult> getAliasingOpResult(Operation* op,
                                              OpOperand& opOperand,
                                              const AnalysisState& state) const {
        auto applyOp = mlir::dyn_cast<ApplyOp>(op);
        auto outputs = applyOp.getOutputs();
        for (size_t i = 0; i < outputs.size(); ++i) {
            if (opOperand.is(outputs[i])) {
                return { op->getOpResult(i) };
            }
        }
        return {};
    }

    BufferRelation bufferRelation(Operation* op,
                                  OpResult opResult,
                                  const AnalysisState& state) const {
        return BufferRelation::Equivalent;
    }

    LogicalResult bufferize(Operation* op,
                            RewriterBase& rewriter,
                            const BufferizationOptions& options) const {
        auto applyOp = mlir::dyn_cast<ApplyOp>(op);

        const auto& inputMemrefsOrFailure = BufferizeValueRange(rewriter, applyOp.getInputs(), options);
        if (failed(inputMemrefsOrFailure)) {
            return inputMemrefsOrFailure;
        }
        const auto& inputMemrefs = *inputMemrefsOrFailure;

        const auto& outputMemrefsOrFailure = BufferizeValueRange(rewriter, applyOp.getOutputs(), options);
        if (failed(outputMemrefsOrFailure)) {
            return outputMemrefsOrFailure;
        }
        const auto& outputMemrefs = *outputMemrefsOrFailure;

        std::vector<int64_t> staticOffsets;
        for (auto offset : applyOp.getStaticOffsets().getAsRange<IntegerAttr>()) {
            staticOffsets.push_back(offset.getInt());
        }

        rewriter.create<ApplyOp>(applyOp->getLoc(),
                                 applyOp.getCallee(),
                                 inputMemrefs,
                                 outputMemrefs,
                                 applyOp.getOffsets(),
                                 staticOffsets);
        replaceOpWithBufferizedValues(rewriter, applyOp, outputMemrefs);

        return success();
    }
};


struct StencilOpInterface : public BufferizableOpInterface::ExternalModel<StencilOpInterface, StencilOp> {
    bool bufferizesToMemoryRead(Operation*, OpOperand&, const AnalysisState&) const {
        return true;
    }

    bool bufferizesToMemoryWrite(Operation*, OpOperand&, const AnalysisState&) const {
        return false;
    }

    SmallVector<OpResult> getAliasingOpResult(Operation*, OpOperand&, const AnalysisState&) const {
        return {};
    }

    BufferRelation bufferRelation(Operation*, OpResult, const AnalysisState&) const {
        return BufferRelation::None;
    }

    LogicalResult bufferize(Operation* op,
                            RewriterBase& rewriter,
                            const BufferizationOptions& options) const {
        auto stencilOp = mlir::dyn_cast<StencilOp>(op);

        // I stole this code from MLIR's bufferization for func::FuncOp.

        // Construct the bufferized function type.
        FunctionType funcType = stencilOp.getFunctionType();
        SmallVector<Type> argTypes;
        for (const auto& it : llvm::enumerate(funcType.getInputs())) {
            Type argType = it.value();
            if (auto tensorType = argType.dyn_cast<TensorType>()) {
                Type memrefType = bufferization::getMemRefTypeWithFullyDynamicLayout(tensorType);
                argTypes.push_back(memrefType);
            }
            else {
                argTypes.push_back(argType);
            }
        }

        // 1. Rewrite the bbArgs. Turn every tensor bbArg into a memref bbArg.
        Block& frontBlock = stencilOp.getBody().front();
        for (BlockArgument& bbArg : frontBlock.getArguments()) {
            auto tensorType = bbArg.getType().dyn_cast<TensorType>();
            // Non-tensor types stay the same.
            if (!tensorType)
                continue;

            // Collect all uses of the bbArg.
            SmallVector<OpOperand*> bbArgUses;
            for (OpOperand& use : bbArg.getUses())
                bbArgUses.push_back(&use);

            // Change the bbArg type to memref.
            Type memrefType = bufferization::getMemRefTypeWithFullyDynamicLayout(tensorType);
            bbArg.setType(memrefType);

            // Replace all uses of the original tensor bbArg.
            rewriter.setInsertionPointToStart(&frontBlock);
            if (!bbArgUses.empty()) {
                // Insert to_tensor because the remaining function body has not been
                // bufferized yet.
                Value toTensorOp = rewriter.create<bufferization::ToTensorOp>(stencilOp.getLoc(), bbArg);
                for (OpOperand* use : bbArgUses)
                    use->set(toTensorOp);
            }
        }

        // 4. Rewrite the FuncOp type to buffer form.
        stencilOp.setType(FunctionType::get(op->getContext(),
                                            argTypes,
                                            funcType.getResults()));

        return success();
    }
};


using InvokeStencilOpInterface = TrivialOpInterface<InvokeStencilOp>;



/// Helper function for loop bufferization. Return the indices of all values
/// that have a tensor type.
static DenseSet<int64_t> getTensorIndices(ValueRange values) {
    DenseSet<int64_t> result;
    for (const auto& it : llvm::enumerate(values))
        if (it.value().getType().isa<TensorType>())
            result.insert(it.index());
    return result;
}

/// Helper function for loop bufferization. Return the indices of all
/// bbArg/yielded value pairs who's buffer relation is "Equivalent".
DenseSet<int64_t> getEquivalentBuffers(Block::BlockArgListType bbArgs,
                                       ValueRange yieldedValues,
                                       const AnalysisState& state) {
    unsigned int minSize = std::min(bbArgs.size(), yieldedValues.size());
    DenseSet<int64_t> result;
    for (unsigned int i = 0; i < minSize; ++i) {
        if (!bbArgs[i].getType().isa<TensorType>() || !yieldedValues[i].getType().isa<TensorType>())
            continue;
        if (state.areEquivalentBufferizedValues(bbArgs[i], yieldedValues[i]))
            result.insert(i);
    }
    return result;
}

/// Helper function for loop bufferization. Given a list of bbArgs of the new
/// (bufferized) loop op, wrap the bufferized tensor args (now memrefs) into
/// ToTensorOps, so that the block body can be moved over to the new op.
SmallVector<Value>
getBbArgReplacements(RewriterBase& rewriter,
                     Block::BlockArgListType bbArgs,
                     const DenseSet<int64_t>& tensorIndices) {
    SmallVector<Value> result;
    for (const auto& it : llvm::enumerate(bbArgs)) {
        size_t idx = it.index();
        Value val = it.value();
        if (tensorIndices.contains(idx)) {
            result.push_back(
                rewriter.create<bufferization::ToTensorOp>(val.getLoc(), val)
                    .getResult());
        }
        else {
            result.push_back(val);
        }
    }
    return result;
}


/// Helper function for loop bufferization. Return the bufferized values of the
/// given OpOperands. If an operand is not a tensor, return the original value.
static FailureOr<SmallVector<Value>>
getBuffers(RewriterBase& rewriter, MutableArrayRef<OpOperand> operands,
           const BufferizationOptions& options) {
    SmallVector<Value> result;
    for (OpOperand& opOperand : operands) {
        if (opOperand.get().getType().isa<TensorType>()) {
            FailureOr<Value> resultBuffer =
                getBuffer(rewriter, opOperand.get(), options);
            if (failed(resultBuffer))
                return failure();
            result.push_back(*resultBuffer);
        }
        else {
            result.push_back(opOperand.get());
        }
    }
    return result;
}

/// Copied from bufferization of scf.for.
struct ForeachElementOpInterface : public BufferizableOpInterface::ExternalModel<ForeachElementOpInterface,
                                                                                 ForeachElementOp> {
    bool bufferizesToMemoryRead(Operation* op, OpOperand& opOperand,
                                const AnalysisState& state) const {
        if (opOperand.getOperandNumber() == 0) {
            return false; // The field over which to iterate is never read, only its size is taken.
        }
        // ForeachElementOp alone doesn't bufferize to a memory read, one of the uses of
        // its matching bbArg may.
        auto forOp = cast<ForeachElementOp>(op);
        return state.isValueRead(forOp.getRegionIterArgForOpOperand(opOperand));
    }

    bool bufferizesToMemoryWrite(Operation* op, OpOperand& opOperand,
                                 const AnalysisState& state) const {
        if (opOperand.getOperandNumber() == 0) {
            return false; // The field over which to iterate is never written, only its size is taken.
        }
        // Tensor iter_args of ForeachElementOps are always considered as a write.
        return true;
    }

    SmallVector<OpResult> getAliasingOpResult(Operation* op, OpOperand& opOperand,
                                              const AnalysisState& state) const {
        if (opOperand.getOperandNumber() == 0) {
            return {}; // There is no result associated with the field.
        }
        auto forOp = cast<ForeachElementOp>(op);
        return { forOp.getResultForOpOperand(opOperand) };
    }

    BufferRelation bufferRelation(Operation* op, OpResult opResult,
                                  const AnalysisState& state) const {
        // ForOp results are equivalent to their corresponding init_args if the
        // corresponding iter_args and yield values are equivalent.
        auto forOp = cast<ForeachElementOp>(op);
        OpOperand& forOperand = forOp.getOpOperandForResult(opResult);
        auto bbArg = forOp.getRegionIterArgForOpOperand(forOperand);
        auto yieldOp = cast<YieldOp>(forOp.getLoopBody().front().getTerminator());
        bool equivalentYield = state.areEquivalentBufferizedValues(
            bbArg, yieldOp->getOperand(opResult.getResultNumber()));
        return equivalentYield ? BufferRelation::Equivalent : BufferRelation::None;
    }

    bool isWritable(Operation* op, Value value,
                    const AnalysisState& state) const {
        // Interestingly, ForeachElementOp's bbArg can **always** be viewed
        // inplace from the perspective of ops nested under:
        //   1. Either the matching iter operand is not bufferized inplace and an
        //      alloc + optional copy makes the bbArg itself inplaceable.
        //   2. Or the matching iter operand is bufferized inplace and bbArg just
        //      bufferizes to that too.
        return true;
    }

    LogicalResult resolveConflicts(Operation* op, RewriterBase& rewriter,
                                   const AnalysisState& state) const {
        auto bufferizableOp = cast<BufferizableOpInterface>(op);
        if (failed(bufferizableOp.resolveTensorOpOperandConflicts(rewriter, state)))
            return failure();

        if (!state.getOptions().enforceAliasingInvariants)
            return success();

        // According to the `getAliasing...` implementations, a bufferized OpResult
        // may alias only with the corresponding bufferized init_arg and with no
        // other buffers. I.e., the i-th OpResult may alias with the i-th init_arg;
        // but not with any other OpOperand. If a corresponding OpResult/init_arg
        // pair bufferizes to equivalent buffers, this aliasing requirement is
        // satisfied. Otherwise, we cannot be sure and must yield a new buffer copy.
        // (New buffer copies do not alias with any buffer.)
        auto forOp = cast<ForeachElementOp>(op);
        auto yieldOp = cast<YieldOp>(forOp.getLoopBody().front().getTerminator());
        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPoint(yieldOp);

        // Indices of all iter_args that have tensor type. These are the ones that
        // are bufferized.
        DenseSet<int64_t> indices = getTensorIndices(forOp.getInitArgs());
        // For every yielded value, is the value equivalent to its corresponding
        // bbArg?
        DenseSet<int64_t> equivalentYields = getEquivalentBuffers(forOp.getRegionIterArgs(),
                                                                  yieldOp.getResults(),
                                                                  state);
        SmallVector<Value> yieldValues;
        for (int64_t idx = 0;
             idx < static_cast<int64_t>(yieldOp.getResults().size()); ++idx) {
            Value value = yieldOp.getResults()[idx];
            if (!indices.contains(idx) || equivalentYields.contains(idx)) {
                yieldValues.push_back(value);
                continue;
            }
            FailureOr<Value> alloc = allocateTensorForShapedValue(rewriter,
                                                                  yieldOp.getLoc(),
                                                                  value,
                                                                  /*escape=*/true,
                                                                  state.getOptions());
            if (failed(alloc))
                return failure();
            yieldValues.push_back(*alloc);
        }

        rewriter.updateRootInPlace(
            yieldOp, [&]() { yieldOp.getResultsMutable().assign(yieldValues); });
        return success();
    }

    FailureOr<BaseMemRefType>
    getBufferType(Operation* op, BlockArgument bbArg,
                  const BufferizationOptions& options) const {
        auto forOp = cast<ForeachElementOp>(op);
        return bufferization::getBufferType(forOp.getOpOperandForRegionIterArg(bbArg).get(), options);
    }

    LogicalResult bufferize(Operation* op, RewriterBase& rewriter,
                            const BufferizationOptions& options) const {
        auto forOp = cast<ForeachElementOp>(op);
        Block* oldLoopBody = &forOp.getLoopBody().front();

        // Indices of all iter_args that have tensor type. These are the ones that
        // are bufferized.
        DenseSet<int64_t> indices = getTensorIndices(forOp.getInitArgs());

        // The new memref init_args of the loop.
        FailureOr<SmallVector<Value>> maybeInitArgs = getBuffers(rewriter, forOp.getIterOpOperands(), options);
        if (failed(maybeInitArgs))
            return failure();
        SmallVector<Value> initArgs = *maybeInitArgs;

        // Construct a new scf.for op with memref instead of tensor values.
        auto maybeBufferizedField = bufferization::getBuffer(rewriter, forOp.getField(), options);
        if (failed(maybeBufferizedField)) {
            return failure();
        }
        auto newForOp = rewriter.create<ForeachElementOp>(forOp.getLoc(),
                                                          maybeBufferizedField.value(),
                                                          forOp.getDimAttr(),
                                                          initArgs);
        newForOp->setAttrs(forOp->getAttrs());
        ValueRange initArgsRange(initArgs);
        TypeRange initArgsTypes(initArgsRange);
        Block* loopBody = &newForOp.getLoopBody().front();

        // Set up new iter_args. The loop body uses tensors, so wrap the (memref)
        // iter_args of the new loop in ToTensorOps.
        rewriter.setInsertionPointToStart(loopBody);
        SmallVector<Value> iterArgs = getBbArgReplacements(rewriter, newForOp.getRegionIterArgs(), indices);
        iterArgs.insert(iterArgs.begin(), newForOp.getInductionVar());

        // Move loop body to new loop.
        rewriter.mergeBlocks(oldLoopBody, loopBody, iterArgs);

        // Replace loop results.
        replaceOpWithBufferizedValues(rewriter, op, newForOp->getResults());

        return success();
    }

    /// Assert that yielded values of an scf.for op are equivalent to their
    /// corresponding bbArgs. In that case, the buffer relations of the
    /// corresponding OpResults are "Equivalent".
    ///
    /// If this is not the case, an allocs+copies are inserted and yielded from
    /// the loop. This could be a performance problem, so it must be explicitly
    /// activated with `alloc-return-allocs`.
    LogicalResult verifyAnalysis(Operation* op, const AnalysisState& state) const {
        const auto& options = static_cast<const OneShotBufferizationOptions&>(state.getOptions());
        if (options.allowReturnAllocs) {
            return success();
        }

        auto forOp = cast<ForeachElementOp>(op);
        auto yieldOp = cast<YieldOp>(forOp.getLoopBody().front().getTerminator());
        for (OpResult opResult : op->getOpResults()) {
            if (!opResult.getType().isa<TensorType>())
                continue;

            // Note: This is overly strict. We should check for aliasing bufferized
            // values. But we don't have a "must-alias" analysis yet.
            if (bufferRelation(op, opResult, state) != BufferRelation::Equivalent)
                return yieldOp->emitError()
                       << "Yield operand #" << opResult.getResultNumber()
                       << " is not equivalent to the corresponding iter bbArg";
        }

        return success();
    }
};


void registerBufferizableOpInterfaceExternalModels(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](MLIRContext* context, StencilDialect* dialect) {
        ApplyOp::attachInterface<ApplyOpInterface>(*context);

        StencilOp::attachInterface<StencilOpInterface>(*context);
        InvokeStencilOp::attachInterface<InvokeStencilOpInterface>(*context);

        SampleOp::attachInterface<TrivialOpInterface<SampleOp>>(*context);
        SampleIndirectOp::attachInterface<TrivialOpInterface<SampleIndirectOp>>(*context);
        JumpIndirectOp::attachInterface<TrivialOpInterface<JumpIndirectOp>>(*context);
        ForeachElementOp::attachInterface<ForeachElementOpInterface>(*context);
    });
}


} // namespace stencil