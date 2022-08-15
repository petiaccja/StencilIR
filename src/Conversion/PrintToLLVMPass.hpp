#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>


class PrintToLLVMPass : public mlir::PassWrapper<PrintToLLVMPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
    void getDependentDialects(mlir::DialectRegistry& registry) const override;
    void runOnOperation() override final;
};