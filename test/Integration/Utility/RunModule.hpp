#pragma once


#include <Compiler/Pipelines.hpp>
#include <Diagnostics/Exception.hpp>
#include <Execution/Execution.hpp>
#include <IR/ConvertOps.hpp>
#include <IR/Ops.hpp>

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <filesystem>
#include <fstream>


template <class... Args>
std::vector<sir::StageResult> RunModule(const sir::ops::ModuleOp& mod,
                                        std::string_view function,
                                        bool optimize,
                                        Args&&... args) {
    mlir::MLIRContext context;
    mlir::ModuleOp ir = mlir::dyn_cast<mlir::ModuleOp>(ConvertOperation(context, mod));

    mlir::PassManager snapshotPm(ir->getContext());
    auto snapshotFile = std::filesystem::temp_directory_path() / "0_0_original.mlir";
    snapshotPm.addPass(mlir::createLocationSnapshotPass({}, snapshotFile.c_str()));
    if (failed(snapshotPm.run(ir))) {
        throw sir::Exception("failed to snapshot IR locations");
    }

    const auto optimizationOptions = !optimize ? sir::OptimizationOptions{} : sir::OptimizationOptions{
        .inlineFunctions = true,
        .fuseExtractSliceOps = true,
        .fuseApplyOps = true,
        .eliminateAllocBuffers = true,
    };

    sir::Compiler compiler{ TargetCPUPipeline(context, optimizationOptions) };
    std::vector<sir::StageResult> stageResults;
    auto writeStageResults = [&]() {
        for (auto& stage : stageResults) {
            const auto outPath = std::filesystem::temp_directory_path() / (stage.name + ".mlir");
            std::ofstream outFile{ outPath, std::ios::trunc };
            outFile << stage.ir;
        }
    };

    try {
        mlir::ModuleOp compiled = compiler.Run(ir, stageResults);
        constexpr int optLevel = 3;
        sir::Runner jitRunner{ compiled, optLevel };
        jitRunner.Invoke(function, std::forward<Args>(args)...);
    }
    catch (...) {
        writeStageResults();
        throw;
    }
    writeStageResults();

    return stageResults;
}