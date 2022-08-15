#include "Lowering.hpp"

#include "LoweringPasses.hpp"
#include "PrintToLLVMPass.hpp"
#include "StencilLoweringPasses.hpp"

#include <StencilDialect/StencilDialect.hpp>

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/LocationSnapshot.h>
#include <mlir/Transforms/Passes.h>

#include <filesystem>


void ThrowIfFailed(mlir::LogicalResult result, std::string msg) {
    if (failed(result)) {
        throw std::runtime_error(std::move(msg));
    }
}


mlir::ModuleOp CloneModule(mlir::ModuleOp original) {
    return mlir::dyn_cast<mlir::ModuleOp>(original->clone());
    ;
}


void ApplyLowerToSCFLoops(mlir::MLIRContext& context, mlir::ModuleOp& op) {
    mlir::PassManager passManager(&context);
    passManager.addPass(std::make_unique<StencilToSCFPass>());
    ThrowIfFailed(passManager.run(op), "Failed to lower to Affine.");
}


void ApplyLowerToFunc(mlir::MLIRContext& context, mlir::ModuleOp& op) {
    mlir::PassManager passManager(&context);
    passManager.addPass(std::make_unique<StencilToFuncPass>());
    passManager.addPass(std::make_unique<PrintToLLVMPass>());
    ThrowIfFailed(passManager.run(op), "Failed to lower to Func.");
}

void ApplyLowerToCf(mlir::MLIRContext& context, mlir::ModuleOp& op) {
    mlir::PassManager passManager(&context);
    passManager.addPass(std::make_unique<ScfToCfPass>());
    ThrowIfFailed(passManager.run(op), "Failed to lower to ControlFlow.");
}


void ApplyLowerToLLVM(mlir::MLIRContext& context, mlir::ModuleOp& op) {
    mlir::PassManager passManager(&context);
    passManager.addPass(std::make_unique<StdToLLVMPass>());
    ThrowIfFailed(passManager.run(op), "Failed to lower to LLVM IR.");
}


void ApplyCleanupPasses(mlir::MLIRContext& context, mlir::ModuleOp& op) {
    mlir::PassManager passManager(&context);
    passManager.addPass(mlir::createCSEPass());
    passManager.addPass(mlir::createCanonicalizerPass());
    passManager.addPass(mlir::createTopologicalSortPass());
    passManager.addPass(mlir::createSCCPPass());
    ThrowIfFailed(passManager.run(op), "Failed to clean up.");
}


void ApplyLocationSnapshot(mlir::MLIRContext& context, mlir::ModuleOp& op) {
    const auto tempPath = std::filesystem::temp_directory_path();
    const auto tempFile = tempPath / "stencil.mlir";
    const auto fileName = tempFile.string();

    mlir::PassManager passManager(&context);
    passManager.addPass(mlir::createLocationSnapshotPass({}, fileName));
    ThrowIfFailed(passManager.run(op), "Failed to snapshot locations.");
}

auto LowerToLLVMCPU(mlir::MLIRContext& context, const mlir::ModuleOp& module)
    -> std::vector<std::pair<std::string, mlir::ModuleOp>> {
    std::vector<std::pair<std::string, mlir::ModuleOp>> stages;
    auto mutableModule = CloneModule(module);

    ApplyLowerToSCFLoops(context, mutableModule);
    ApplyCleanupPasses(context, mutableModule);
    stages.push_back({ "Loops", CloneModule(mutableModule) });

    ApplyLowerToFunc(context, mutableModule);
    ApplyCleanupPasses(context, mutableModule);
    stages.push_back({ "Func", CloneModule(mutableModule) });

    ApplyLowerToCf(context, mutableModule);
    ApplyCleanupPasses(context, mutableModule);
    stages.push_back({ "ControlFlow", CloneModule(mutableModule) });

    ApplyLowerToLLVM(context, mutableModule);
    ApplyCleanupPasses(context, mutableModule);
    stages.push_back({ "MLIR/LLVM", CloneModule(mutableModule) });

    return stages;
}

void ApplyLowerToGPU(mlir::MLIRContext& context, mlir::ModuleOp& op) {
    mlir::PassManager passManager(&context);
    passManager.addPass(std::make_unique<StencilToGPUPass>());
    ThrowIfFailed(passManager.run(op), "Failed to lower to GPU.");
}

#include <mlir/Conversion/GPUToVulkan/ConvertGPUToVulkanPass.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>

auto LowerToLLVMGPU(mlir::MLIRContext& context, const mlir::ModuleOp& module)
    -> std::vector<std::pair<std::string, mlir::ModuleOp>> {
    std::vector<std::pair<std::string, mlir::ModuleOp>> stages;
    auto mutableModule = CloneModule(module);

    ApplyLowerToGPU(context, mutableModule);
    ApplyCleanupPasses(context, mutableModule);
    stages.push_back({ "GPU", CloneModule(mutableModule) });

    mlir::PassManager pm(&context);
    pm.addPass(mlir::createGpuKernelOutliningPass());
    
    //pm.addPass(mlir::createConvertGpuLaunchFuncToVulkanLaunchFuncPass());
    ThrowIfFailed(pm.run(mutableModule), "Could not lower GPU to Vulkan launch");
    stages.push_back({ "Vulkan", CloneModule(mutableModule) });

    return stages;
}
