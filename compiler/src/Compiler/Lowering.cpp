#include "Lowering.hpp"

#include "KernelToAffinePass.hpp"
#include "LoweringPasses.hpp"
#include "MockPrintPass.hpp"

#include <MockDialect/MockDialect.hpp>

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


void ApplyLowerToAffine(mlir::MLIRContext& context, mlir::ModuleOp& op, bool makeParallelLoops = false) {
    mlir::PassManager passManager(&context);
    passManager.addPass(std::make_unique<KernelToAffinePass>(makeParallelLoops));
    passManager.addPass(std::make_unique<MockPrintPass>());
    ThrowIfFailed(passManager.run(op), "Failed to lower to Affine.");
}


void ApplyLowerToScf(mlir::MLIRContext& context, mlir::ModuleOp& op) {
    mlir::PassManager passManager(&context);
    passManager.addPass(std::make_unique<AffineToScfPass>());
    ThrowIfFailed(passManager.run(op), "Failed to lower to SCF.");
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
    ThrowIfFailed(passManager.run(op), "Failed to clean up.");
}


void ApplyLocationSnapshot(mlir::MLIRContext& context, mlir::ModuleOp& op) {
    const auto tempPath = std::filesystem::temp_directory_path();
    const auto tempFile = tempPath / "mock.mlir";
    const auto fileName = tempFile.string();

    mlir::PassManager passManager(&context);
    passManager.addPass(mlir::createLocationSnapshotPass({}, fileName));
    ThrowIfFailed(passManager.run(op), "Failed to snapshot locations.");
}

auto LowerToLLVMCPU(mlir::MLIRContext& context, const mlir::ModuleOp& module)
    -> std::vector<std::pair<std::string, mlir::ModuleOp>> {
    std::vector<std::pair<std::string, mlir::ModuleOp>> stages;
    auto mutableModule = CloneModule(module);

    ApplyLowerToAffine(context, mutableModule);
    ApplyCleanupPasses(context, mutableModule);
    stages.push_back({ "Affine & Func", CloneModule(mutableModule) });

    ApplyLowerToScf(context, mutableModule);
    ApplyCleanupPasses(context, mutableModule);
    stages.push_back({ "SCF", CloneModule(mutableModule) });

    ApplyLowerToCf(context, mutableModule);
    ApplyCleanupPasses(context, mutableModule);
    stages.push_back({ "ControlFlow", CloneModule(mutableModule) });

    ApplyLowerToLLVM(context, mutableModule);
    ApplyCleanupPasses(context, mutableModule);
    stages.push_back({ "LLVM", CloneModule(mutableModule) });

    return stages;
}


auto LowerToLLVMGPU(mlir::MLIRContext& context, const mlir::ModuleOp& module) {
    std::vector<std::pair<std::string, std::string>> stages;
    auto clone = mlir::dyn_cast<mlir::ModuleOp>(module->clone());

    return std::tuple{ clone, stages };
}
