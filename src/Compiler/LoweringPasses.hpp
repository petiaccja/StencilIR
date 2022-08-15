#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>


class ScfToCfPass : public mlir::PassWrapper<ScfToCfPass , mlir::OperationPass<mlir::ModuleOp>> {
    void getDependentDialects(mlir::DialectRegistry& registry) const override;
    void runOnOperation() override final;
};

class StdToLLVMPass : public mlir::PassWrapper<StdToLLVMPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
    void getDependentDialects(mlir::DialectRegistry& registry) const override;
    void runOnOperation() override final;
};