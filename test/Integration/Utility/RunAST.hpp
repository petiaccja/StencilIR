#pragma once


#include <AST/ConvertASTToIR.hpp>
#include <AST/Nodes.hpp>
#include <Compiler/Pipelines.hpp>
#include <Execution/Execution.hpp>


template <class... Args>
std::vector<StageResult> RunAST(const ast::Module& ast, std::string_view function, bool optimize, Args&&... args) {
    mlir::MLIRContext context;
    mlir::ModuleOp ir = ConvertASTToIR(context, ast);

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