#pragma once

#include <Dialect/Stencil/IR/StencilOps.hpp>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>


mlir::FailureOr<stencil::ApplyOp> DeduplicateApplyInputs(stencil::ApplyOp applyOp, mlir::PatternRewriter& rewriter);

class DeduplicateApplyInputsPass : public mlir::PassWrapper<DeduplicateApplyInputsPass, mlir::OperationPass<mlir::ModuleOp>> {
    void runOnOperation() override final;
};
