#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>


class StencilToStdPass : public mlir::PassWrapper<StencilToStdPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
    void getDependentDialects(mlir::DialectRegistry& registry) const override;
    void runOnOperation() override final;
    bool launchToGpu = false;
};