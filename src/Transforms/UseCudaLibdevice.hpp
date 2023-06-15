#pragma once

#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Pass/Pass.h>


namespace sir {

class UseCudaLibdevicePass : public mlir::PassWrapper<UseCudaLibdevicePass, mlir::OperationPass<mlir::gpu::GPUModuleOp>> {
    void runOnOperation() override final;
};


} // namespace sir