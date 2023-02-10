#include "FuseExtractSliceOps.hpp"

#include "DeduplicateApplyInputs.hpp"
#include "Utility.hpp"

#include <Dialect/Stencil/IR/StencilOps.hpp>

#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include <algorithm>
#include <optional>
#include <ranges>
#include <regex>
#include <span>

using mlir::MLIRContext;


auto GetStaticOffsets(mlir::tensor::ExtractSliceOp extractSliceOp)
    -> std::optional<mlir::SmallVector<int64_t, 3>> {
    const auto rank = extractSliceOp->getResultTypes()[0].cast<mlir::ShapedType>().getRank();
    const auto staticOffsets = extractSliceOp.getStaticOffsets().getAsRange<mlir::IntegerAttr>();
    const auto staticStrides = extractSliceOp.getStaticStrides().getAsRange<mlir::IntegerAttr>();

    mlir::SmallVector<int64_t, 3> offsets;
    for (auto offset : staticOffsets) {
        offsets.push_back(offset.getInt());
    }

    const bool allOffsetsStatic = std::none_of(offsets.begin(), offsets.end(), [](int64_t offset) {
                                      return offset == mlir::ShapedType::kDynamicStrideOrOffset;
                                  })
                                  && offsets.size() == size_t(rank);

    const bool allStridesOne = std::all_of(staticStrides.begin(), staticStrides.end(), [](mlir::IntegerAttr stride) {
                                   return stride.getInt() == 1;
                               })
                               && std::distance(staticStrides.begin(), staticStrides.end()) == rank;

    if (allOffsetsStatic && allStridesOne) {
        return offsets;
    }
    return {};
}


auto OffsetStencilInputs(stencil::StencilOp stencilOp,
                         std::span<std::optional<mlir::SmallVector<int64_t, 3>>> offsets,
                         mlir::PatternRewriter& rewriter)
    -> mlir::FailureOr<stencil::StencilOp> {
    rewriter.setInsertionPointAfter(stencilOp);
    auto offsetedStencil = mlir::dyn_cast<stencil::StencilOp>(rewriter.clone(*stencilOp));
    auto blockArgs = offsetedStencil.getRegion().getBlocks().front().getArguments();
    assert(blockArgs.size() == offsets.size());

    auto offsetIt = offsets.begin();
    auto blockArgIt = blockArgs.begin();
    for (; offsetIt != offsets.end(); ++offsetIt, ++blockArgIt) {
        if (!*offsetIt) {
            continue;
        }
        auto blockArg = *blockArgIt;
        for (auto userOp : blockArg.getUsers()) {
            if (auto sampleOp = mlir::dyn_cast<stencil::SampleOp>(userOp)) {
                const auto& offset = offsetIt->value();
                auto originalIndex = sampleOp.getIndex();
                rewriter.setInsertionPointAfterValue(originalIndex);
                auto offsetedIndex = rewriter.create<stencil::JumpOp>(sampleOp->getLoc(),
                                                                      originalIndex.getType(),
                                                                      originalIndex,
                                                                      rewriter.getI64ArrayAttr(offset));
                sampleOp.getIndexMutable().assign(offsetedIndex);
            }
            else {
                rewriter.eraseOp(offsetedStencil);
                return mlir::failure();
            }
        }
    }
    offsetedStencil.setSymNameAttr(UniqueStencilName(stencilOp, "offseted", rewriter));
    return offsetedStencil;
}


mlir::FailureOr<stencil::ApplyOp> FusePrecedingExtractSlices(stencil::ApplyOp applyOp, mlir::PatternRewriter& rewriter) {
    mlir::SmallVector<std::optional<mlir::SmallVector<int64_t, 3>>, 16> offsets;
    for (const auto& input : applyOp.getInputs()) {
        const auto definingOp = input.getDefiningOp();
        if (definingOp) {
            if (auto extractSliceOp = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(definingOp)) {
                offsets.push_back(GetStaticOffsets(extractSliceOp));
            }
        }
    }

    if (std::none_of(offsets.begin(), offsets.end(), std::identity{})) {
        return applyOp;
    }

    auto stencilOp = mlir::cast<stencil::StencilOp>(mlir::SymbolTable::lookupNearestSymbolFrom(applyOp, applyOp.getCalleeAttr()));
    auto maybeOffsetedStencilOp = OffsetStencilInputs(stencilOp, offsets, rewriter);
    if (failed(maybeOffsetedStencilOp)) {
        return mlir::failure();
    }

    auto offsetedStencilOp = maybeOffsetedStencilOp.value();
    rewriter.setInsertionPointAfter(applyOp);
    auto offsetedApplyOp = mlir::cast<stencil::ApplyOp>(rewriter.clone(*applyOp));

    auto inputs = offsetedApplyOp.getInputs();
    assert(inputs.size() == offsets.size());

    mlir::SmallVector<mlir::Value, 16> offsetedInputs;
    offsetedInputs.reserve(inputs.size());
    auto inputIt = inputs.begin();
    auto offsetIt = offsets.begin();
    for (; inputIt != inputs.end(); ++inputIt, ++offsetIt) {
        auto input = *inputIt;
        if (*offsetIt) {
            auto extractSliceOp = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(input.getDefiningOp());
            assert(extractSliceOp);
            offsetedInputs.push_back(extractSliceOp.getSource());
        }
        else {
            offsetedInputs.push_back(input);
        }
    }
    offsetedApplyOp.setCalleeAttr(mlir::FlatSymbolRefAttr::get(offsetedStencilOp.getSymNameAttr()));
    offsetedApplyOp.getInputsMutable().assign(offsetedInputs);

    auto dedupApplyOp = DeduplicateApplyInputs(offsetedApplyOp, rewriter);
    if (succeeded(dedupApplyOp)) {
        rewriter.eraseOp(offsetedApplyOp);
        rewriter.eraseOp(offsetedStencilOp);
        return dedupApplyOp;
    }
    return offsetedApplyOp;
}


class FuseExtractSliceOpsPattern : public mlir::OpRewritePattern<stencil::ApplyOp> {
public:
    FuseExtractSliceOpsPattern(MLIRContext* context,
                               mlir::PatternBenefit benefit = 1)
        : OpRewritePattern<stencil::ApplyOp>(context, benefit) {}

    mlir::LogicalResult matchAndRewrite(stencil::ApplyOp applyOp,
                                        mlir::PatternRewriter& rewriter) const override {
        auto maybeFusedOp = FusePrecedingExtractSlices(applyOp, rewriter);
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


void FuseExtractSliceOpsPass::runOnOperation() {
    mlir::Operation* op = getOperation();
    MLIRContext* context = op->getContext();

    mlir::RewritePatternSet patterns(context);
    patterns.add<FuseExtractSliceOpsPattern>(context);

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