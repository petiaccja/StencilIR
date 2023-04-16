#include "ReduceDimOps.hpp"

#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>


namespace sir {

using mlir::MLIRContext;


mlir::FailureOr<mlir::Value> GetEquivalentBuffer(mlir::Value buffer, mlir::bufferization::AnalysisState& state) {
    const auto definingOp = buffer.getDefiningOp();
    if (!definingOp) {
        return mlir::failure();
    }
    auto definingBufferOp = mlir::dyn_cast<mlir::bufferization::BufferizableOpInterface>(definingOp);
    if (!definingBufferOp) {
        return mlir::failure();
    }
    const auto results = definingBufferOp->getResults();
    const auto resultIt = std::find(results.begin(), results.end(), buffer);
    const auto resultIdx = std::distance(results.begin(), resultIt);
    const auto opResult = definingBufferOp->getOpResult(resultIdx);

    const auto aliasingOperands = state.getAliasingOpOperand(opResult);
    if (aliasingOperands.empty()) {
        return mlir::failure();
    }
    if (definingBufferOp.bufferRelation(opResult, state) != mlir::bufferization::BufferRelation::Equivalent) {
        return mlir::failure();
    }
    const auto equivalentBuffer = (*aliasingOperands.begin())->get();
    return equivalentBuffer;
}


class ReducePattern : public mlir::OpRewritePattern<mlir::tensor::DimOp> {
public:
    ReducePattern(MLIRContext* context,
                  mlir::bufferization::AnalysisState& state,
                  mlir::PatternBenefit benefit = 1)
        : OpRewritePattern<mlir::tensor::DimOp>(context, benefit), state(state) {}


    mlir::LogicalResult matchAndRewrite(mlir::tensor::DimOp dimOp,
                                        mlir::PatternRewriter& rewriter) const override {
        const auto source = dimOp.getSource();
        const auto maybeEquivalentSource = GetEquivalentBuffer(source, state);
        if (succeeded(maybeEquivalentSource)) {
            const auto equivalentSource = *maybeEquivalentSource;
            const auto index = dimOp.getIndex();
            const auto operands = mlir::SmallVector<mlir::Value>{ equivalentSource, index };
            const auto attrs = dimOp->getAttrs();
            rewriter.replaceOpWithNewOp<mlir::tensor::DimOp>(dimOp, operands, attrs);
        }
        return mlir::failure();
    }

    mlir::bufferization::AnalysisState& state;
};


void ReduceDimOpsPass::runOnOperation() {
    mlir::Operation* op = getOperation();
    MLIRContext* context = op->getContext();

    // No need for proper bufferization options or analysis.
    // Queried bufferization interface properties should work without.
    const mlir::bufferization::BufferizationOptions bufferizationOptions{};
    mlir::bufferization::AnalysisState state{ bufferizationOptions };

    mlir::RewritePatternSet patterns(context);
    patterns.add<ReducePattern>(context, state);

    // Use TopDownTraversal for compile time reasons
    mlir::GreedyRewriteConfig grc;
    grc.useTopDownTraversal = true;
    (void)applyPatternsAndFoldGreedily(op->getRegions(), std::move(patterns), grc);

    mlir::PassManager pm{ &getContext() };
    pm.addPass(mlir::createCanonicalizerPass());
    if (pm.run(op).failed()) {
        signalPassFailure();
    }
}


} // namespace sir