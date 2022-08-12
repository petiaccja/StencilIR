#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>


class KernelToAffinePass : public mlir::PassWrapper<KernelToAffinePass, mlir::OperationPass<mlir::ModuleOp>> {
public:
    KernelToAffinePass(bool makeParallelLoops = false) : m_makeParallelLoops(makeParallelLoops) {}
    void getDependentDialects(mlir::DialectRegistry& registry) const override;
    void runOnOperation() override final;

private:
    bool m_makeParallelLoops;
};