#pragma once


#include <Compiler/Pipelines.hpp>
#include <Execution/Execution.hpp>
#include <IR/ConvertOps.hpp>
#include <IR/Ops.hpp>


template <class... Args>
std::vector<StageResult> RunModule(const dag::ModuleOp& mod, std::string_view function, bool optimize, Args&&... args) {
    mlir::MLIRContext context;
    mlir::ModuleOp ir = mlir::dyn_cast<mlir::ModuleOp>(dag::ConvertOperation(context, mod));

    const auto optimizationOptions = !optimize ? OptimizationOptions{} : OptimizationOptions{
        .inlineFunctions = true,
        .fuseExtractSliceOps = true,
        .fuseApplyOps = true,
        .eliminateAllocBuffers = true,
    };

    Compiler compiler{ TargetCPUPipeline(context, optimizationOptions) };
    std::vector<StageResult> stageResults;

    mlir::ModuleOp compiled = compiler.Run(ir, stageResults);

    constexpr int optLevel = 3;
    Runner jitRunner{ compiled, optLevel };
    jitRunner.Invoke(function, std::forward<Args>(args)...);

    return stageResults;
}