#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Pass/Pass.h>


namespace sir {

class StencilApplyToLoopsPass : public mlir::PassWrapper<StencilApplyToLoopsPass, mlir::OperationPass<mlir::ModuleOp>> {
    void getDependentDialects(mlir::DialectRegistry& registry) const override;
    void runOnOperation() override final;
};

} // namespace sir