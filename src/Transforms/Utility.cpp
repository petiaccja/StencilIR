#include "Utility.hpp"

#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>


namespace sir {


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


} // namespace sir