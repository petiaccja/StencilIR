#include "EliminateSlicing.hpp"

#include "Utility.hpp"

#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>

#include <algorithm>


namespace sir {

using mlir::MLIRContext;


bool IsCompleteSlice(mlir::ArrayAttr staticOffsets,
                     mlir::ValueRange sizes,
                     mlir::ArrayAttr staticStrides,
                     mlir::ValueRange bufferSizes) {
    auto staticOffsetAttrs = staticOffsets.getAsRange<mlir::IntegerAttr>();
    auto statiStrideAttrs = staticStrides.getAsRange<mlir::IntegerAttr>();
    const bool completeOffsets = std::all_of(staticOffsetAttrs.begin(), staticOffsetAttrs.end(), [](const auto& offset) {
        return offset.getInt() == 0;
    });
    const bool completeStrides = std::all_of(statiStrideAttrs.begin(), statiStrideAttrs.end(), [](const auto& stride) {
        return stride.getInt() == 1;
    });
    const bool equalSizes = sizes == bufferSizes && sizes.size() == staticOffsets.size();
    return completeOffsets && completeStrides && equalSizes;
}

class RemoveExtractAfterAllocPattern : public mlir::OpRewritePattern<mlir::tensor::ExtractSliceOp> {
public:
    RemoveExtractAfterAllocPattern(MLIRContext* context,
                                   mlir::PatternBenefit benefit = 1)
        : OpRewritePattern<mlir::tensor::ExtractSliceOp>(context, benefit) {}


    mlir::LogicalResult matchAndRewrite(mlir::tensor::ExtractSliceOp op,
                                        mlir::PatternRewriter& rewriter) const override {
        auto def = op.getSource().getDefiningOp();
        if (!def) {
            return mlir::failure();
        }
        auto alloc = mlir::dyn_cast<mlir::bufferization::AllocTensorOp>(def);
        if (!alloc) {
            return mlir::failure();
        }

        if (!IsCompleteSlice(op.getStaticOffsets(), op.getSizes(), op.getStaticStrides(), alloc.getDynamicSizes())) {
            return mlir::failure();
        }

        rewriter.replaceOp(op, alloc->getResults());
        return mlir::success();
    }
};


class RemoveInsertAfterAllocPattern : public mlir::OpRewritePattern<mlir::tensor::InsertSliceOp> {
public:
    RemoveInsertAfterAllocPattern(MLIRContext* context,
                                  mlir::bufferization::AnalysisState& state,
                                  mlir::PatternBenefit benefit = 1)
        : OpRewritePattern<mlir::tensor::InsertSliceOp>(context, benefit), state(state) {}


    mlir::LogicalResult matchAndRewrite(mlir::tensor::InsertSliceOp op,
                                        mlir::PatternRewriter& rewriter) const override {
        auto source = op.getSource();
        auto dest = op.getDest();

        auto maybeSourceEquivalent = GetEquivalentBuffer(source, state);
        if (failed(maybeSourceEquivalent)) {
            return mlir::failure();
        }
        auto sourceEquivalent = maybeSourceEquivalent.value();

        auto def = sourceEquivalent.getDefiningOp();
        if (!def) {
            return mlir::failure();
        }
        auto alloc = mlir::dyn_cast<mlir::bufferization::AllocTensorOp>(def);
        if (!alloc) {
            return mlir::failure();
        }

        if (dest != sourceEquivalent) {
            return mlir::failure();
        }
        if (!IsCompleteSlice(op.getStaticOffsets(), op.getSizes(), op.getStaticStrides(), alloc.getDynamicSizes())) {
            return mlir::failure();
        }

        rewriter.replaceOp(op, source.getDefiningOp()->getResults());
        return mlir::success();
    }

    mlir::bufferization::AnalysisState& state;
};


void EliminateSlicingPass::runOnOperation() {
    mlir::Operation* op = getOperation();
    MLIRContext* context = op->getContext();

    // No need for proper bufferization options or analysis.
    // Queried bufferization interface properties should work without.
    const mlir::bufferization::BufferizationOptions bufferizationOptions{};
    mlir::bufferization::AnalysisState state{ bufferizationOptions };

    mlir::RewritePatternSet patterns(context);
    patterns.add<RemoveExtractAfterAllocPattern>(context);
    patterns.add<RemoveInsertAfterAllocPattern>(context, state);

    // Use TopDownTraversal for compile time reasons
    mlir::GreedyRewriteConfig grc;
    grc.useTopDownTraversal = true;
    (void)applyPatternsAndFoldGreedily(op->getRegions(), std::move(patterns), grc);
}


} // namespace sir