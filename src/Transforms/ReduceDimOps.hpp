#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Pass/Pass.h>


namespace sir {

class ReduceDimOpsPass : public mlir::PassWrapper<ReduceDimOpsPass, mlir::OperationPass<mlir::ModuleOp>> {
    void runOnOperation() override final;
};


}