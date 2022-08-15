#pragma once

#include <mlir/IR/BuiltinOps.h>

#include <string>
#include <vector>


void ApplyCleanupPasses(mlir::MLIRContext& context, mlir::ModuleOp& op);

void ApplyLocationSnapshot(mlir::MLIRContext& context, mlir::ModuleOp& op);

auto LowerToLLVMCPU(mlir::MLIRContext& context, const mlir::ModuleOp& module)
    -> std::vector<std::pair<std::string, mlir::ModuleOp>>;

auto LowerToLLVMGPU(mlir::MLIRContext& context, const mlir::ModuleOp& module)
    -> std::vector<std::pair<std::string, mlir::ModuleOp>>;