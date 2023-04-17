#include "EliminateUnusedAllocTensors.hpp"

#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>


namespace sir {

using mlir::MLIRContext;


class EliminatePattern : public mlir::OpRewritePattern<mlir::bufferization::AllocTensorOp> {
public:
    EliminatePattern(MLIRContext* context,
                     mlir::PatternBenefit benefit = 1)
        : OpRewritePattern<mlir::bufferization::AllocTensorOp>(context, benefit) {}


    mlir::LogicalResult matchAndRewrite(mlir::bufferization::AllocTensorOp allocOp,
                                        mlir::PatternRewriter& rewriter) const override {
        const auto buffer = allocOp->getResult(0);
        if (buffer.getUses().empty()) {
            rewriter.eraseOp(allocOp);
            return mlir::success();
        }
        return mlir::failure();
    }
};


void EliminateUnusedAllocTensorsPass::runOnOperation() {
    mlir::Operation* op = getOperation();
    MLIRContext* context = op->getContext();


    mlir::RewritePatternSet patterns(context);
    patterns.add<EliminatePattern>(context);

    // Use TopDownTraversal for compile time reasons
    mlir::GreedyRewriteConfig grc;
    grc.useTopDownTraversal = true;
    (void)applyPatternsAndFoldGreedily(op->getRegions(), std::move(patterns), grc);
}


} // namespace sir