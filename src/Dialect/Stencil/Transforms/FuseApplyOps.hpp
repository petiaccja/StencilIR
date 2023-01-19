#pragma once

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Pass/Pass.h>


class FuseApplyOpsPass : public mlir::PassWrapper<FuseApplyOpsPass, mlir::OperationPass<mlir::ModuleOp>> {
    // void getDependentDialects(mlir::DialectRegistry& registry) const override;
    void runOnOperation() override final;
};
