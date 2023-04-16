#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Pass/Pass.h>


namespace sir {


class StencilToStandardPass : public mlir::PassWrapper<StencilToStandardPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
    void getDependentDialects(mlir::DialectRegistry& registry) const override;
    void runOnOperation() override final;
};


} // namespace sir