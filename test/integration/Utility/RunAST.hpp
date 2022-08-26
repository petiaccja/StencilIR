#pragma once


#include <AST/ASTNodes.hpp>
#include <AST/ConvertASTToIR.hpp>
#include <Compiler/Pipelines.hpp>
#include <Execution/Execution.hpp>


template <class... Args>
std::vector<StageResult> RunAST(const ast::Module& ast, std::string_view function, Args&&... args) {
    mlir::MLIRContext context;
    mlir::ModuleOp ir = ConvertASTToIR(context, ast);

    Compiler compiler{ TargetCPUPipeline(context) };
    std::vector<StageResult> stageResults;

    mlir::ModuleOp compiled = compiler.Run(ir, stageResults);

    constexpr int optLevel = 3;
    JitRunner jitRunner{ compiled, optLevel };
    jitRunner.InvokeFunction(function, std::forward<Args>(args)...);

    return stageResults;
}