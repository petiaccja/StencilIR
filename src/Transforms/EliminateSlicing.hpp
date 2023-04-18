#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Pass/Pass.h>


namespace sir {

class EliminateSlicingPass : public mlir::PassWrapper<EliminateSlicingPass, mlir::OperationPass<>> {
    void runOnOperation() override final;
};


} // namespace sir