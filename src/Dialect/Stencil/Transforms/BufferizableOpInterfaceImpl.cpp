#include "BufferizableOpInterfaceImpl.hpp"

#include "../IR/StencilOps.hpp"

#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>

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


        replaceOpWithNewBufferizedOp<ApplyOp>(rewriter, applyOp,
                                              applyOp.getCallee(),
                                              inputMemrefs,
                                              outputMemrefs,
                                              applyOp.getOffsets(),
                                              staticOffsets);

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


void registerBufferizableOpInterfaceExternalModels(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](MLIRContext* context, StencilDialect* dialect) {
        ApplyOp::attachInterface<ApplyOpInterface>(*context);

        StencilOp::attachInterface<StencilOpInterface>(*context);
        InvokeStencilOp::attachInterface<InvokeStencilOpInterface>(*context);

        SampleOp::attachInterface<TrivialOpInterface<SampleOp>>(*context);
        SampleIndirectOp::attachInterface<TrivialOpInterface<SampleIndirectOp>>(*context);
        JumpIndirectOp::attachInterface<TrivialOpInterface<JumpIndirectOp>>(*context);
    });
}


} // namespace stencil