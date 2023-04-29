#include "DeduplicateApplyInputs.hpp"

#include "Utility.hpp"

#include <Dialect/Stencil/IR/StencilOps.hpp>

#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/InliningUtils.h>
#include <mlir/Transforms/Passes.h>

#include <algorithm>
#include <optional>
#include <ranges>
#include <regex>
#include <span>


using mlir::MLIRContext;


mlir::FailureOr<stencil::StencilOp> DeduplicateStencilInputs(stencil::StencilOp stencilOp,
                                                             std::span<const std::optional<size_t>> replaceWith,
                                                             mlir::PatternRewriter& rewriter) {
    rewriter.setInsertionPointAfter(stencilOp);
    stencil::StencilOp deduplicatedStencil = mlir::dyn_cast<stencil::StencilOp>(rewriter.clone(*stencilOp));
    auto& deduplicatedEntryBlock = deduplicatedStencil.getRegion().front();
    mlir::SmallVector<mlir::Type, 16> deduplicatedParamTypes{ stencilOp.getFunctionType().getInputs().begin(),
                                                              stencilOp.getFunctionType().getInputs().end() };
    const auto deduplicatedResultTypes = deduplicatedStencil.getFunctionType().getResults();

    // Remove duplicate arguments
    auto replaceWithIt = replaceWith.rbegin();
    auto blockArgIdx = deduplicatedEntryBlock.getNumArguments() - 1;
    assert(replaceWith.size() == deduplicatedEntryBlock.getNumArguments());
    for (; replaceWithIt != replaceWith.rend(); ++replaceWithIt, --blockArgIdx) {
        if (*replaceWithIt) {
            const size_t replacementIdx = replaceWithIt->value();
            auto replacedArg = deduplicatedEntryBlock.getArgument(blockArgIdx);
            auto replacementArg = deduplicatedEntryBlock.getArgument(unsigned(replacementIdx));
            replacedArg.replaceAllUsesWith(replacementArg);
            deduplicatedEntryBlock.eraseArgument(blockArgIdx);
            deduplicatedParamTypes.erase(deduplicatedParamTypes.begin() + blockArgIdx);
        }
    }

    // Update function type
    deduplicatedStencil.setFunctionTypeAttr(mlir::TypeAttr::get(rewriter.getFunctionType(deduplicatedParamTypes, deduplicatedResultTypes)));
    deduplicatedStencil.setSymNameAttr(UniqueStencilName(stencilOp, rewriter));
    deduplicatedStencil.setVisibility(mlir::SymbolTable::Visibility::Private);

    return deduplicatedStencil;
}


mlir::FailureOr<stencil::ApplyOp> DeduplicateApplyInputs(stencil::ApplyOp applyOp, mlir::PatternRewriter& rewriter) {
    auto inputs = applyOp.getInputs();
    const size_t numInputs = inputs.size();

    mlir::SmallVector<std::optional<size_t>, 16> replaceWith(numInputs);
    for (size_t inputIdx = 0; inputIdx < numInputs; ++inputIdx) {
        for (size_t candidateIdx = 0; candidateIdx < inputIdx; ++candidateIdx) {
            if (inputs[inputIdx] == inputs[candidateIdx]) {
                replaceWith[inputIdx] = candidateIdx;
            }
        }
    }

    if (!std::any_of(replaceWith.begin(), replaceWith.end(), std::identity{})) {
        return mlir::failure();
    }

    auto stencilOp = mlir::cast<stencil::StencilOp>(mlir::SymbolTable::lookupNearestSymbolFrom(applyOp, applyOp.getCalleeAttr()));
    auto maybeDeduplicatedStencil = DeduplicateStencilInputs(stencilOp, replaceWith, rewriter);
    if (failed(maybeDeduplicatedStencil)) {
        return mlir::failure();
    }

    auto deduplicatedStencil = maybeDeduplicatedStencil.value();
    rewriter.setInsertionPointAfter(applyOp);
    auto deduplicatedApply = mlir::cast<stencil::ApplyOp>(rewriter.clone(*applyOp));

    mlir::SmallVector<mlir::Value, 16> deduplicatedInputs;
    deduplicatedInputs.reserve(numInputs);
    auto inputIt = inputs.begin();
    for (const auto& replacement : replaceWith) {
        if (!replacement) {
            deduplicatedInputs.push_back(*inputIt);
        }
        ++inputIt;
    }

    deduplicatedApply.setCalleeAttr(mlir::FlatSymbolRefAttr::get(deduplicatedStencil.getSymNameAttr()));
    deduplicatedApply.getInputsMutable().assign(deduplicatedInputs);

    return deduplicatedApply;
}


class DeduplicateApplyInputsPattern : public mlir::OpRewritePattern<stencil::ApplyOp> {
public:
    DeduplicateApplyInputsPattern(MLIRContext* context,
                                  mlir::PatternBenefit benefit = 1)
        : OpRewritePattern<stencil::ApplyOp>(context, benefit) {}

    mlir::LogicalResult matchAndRewrite(stencil::ApplyOp applyOp,
                                        mlir::PatternRewriter& rewriter) const override {
        auto maybeFusedOp = DeduplicateApplyInputs(applyOp, rewriter);
        if (succeeded(maybeFusedOp)) {
            auto fusedOp = maybeFusedOp.value();
            if (fusedOp != applyOp) {
                rewriter.replaceOp(applyOp, fusedOp->getResults());
            }
            return mlir::success();
        }
        return mlir::failure();
    }
};


void DeduplicateApplyInputsPass::runOnOperation() {
    mlir::Operation* op = getOperation();
    MLIRContext* context = op->getContext();

    mlir::RewritePatternSet patterns(context);
    patterns.add<DeduplicateApplyInputsPattern>(context);

    // Use TopDownTraversal for compile time reasons
    mlir::GreedyRewriteConfig grc;
    grc.useTopDownTraversal = true;
    (void)applyPatternsAndFoldGreedily(op->getRegions(), std::move(patterns), grc);

    mlir::PassManager pm{ &getContext() };
    pm.addPass(mlir::createSymbolDCEPass());
    if (pm.run(op).failed()) {
        signalPassFailure();
    }
}