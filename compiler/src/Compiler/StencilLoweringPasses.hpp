#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>


class StencilToAffinePass : public mlir::PassWrapper<StencilToAffinePass, mlir::OperationPass<mlir::ModuleOp>> {
public:
    StencilToAffinePass(bool makeParallelLoops = false) : m_makeParallelLoops(makeParallelLoops) {}
    void getDependentDialects(mlir::DialectRegistry& registry) const override;
    void runOnOperation() override final;

private:
    bool m_makeParallelLoops;
};

class StencilToFuncPass : public mlir::PassWrapper<StencilToFuncPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
    void getDependentDialects(mlir::DialectRegistry& registry) const override;
    void runOnOperation() override final;
};