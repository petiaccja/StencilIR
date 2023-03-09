#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Pass/Pass.h>


class FuseExtractSliceOpsPass : public mlir::PassWrapper<FuseExtractSliceOpsPass, mlir::OperationPass<mlir::ModuleOp>> {
    void runOnOperation() override final;
};
