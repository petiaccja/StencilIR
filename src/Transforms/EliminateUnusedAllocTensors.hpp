#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Pass/Pass.h>


namespace sir {

class EliminateUnusedAllocTensorsPass : public mlir::PassWrapper<EliminateUnusedAllocTensorsPass, mlir::OperationPass<mlir::ModuleOp>> {
    void runOnOperation() override final;
};


} // namespace sir