#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <AST/Nodes.hpp>
#include <Compiler/Pipelines.hpp>
#include <DAG/Ops.hpp>
#include <Execution/Execution.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>


enum class eTargetArch {
    X86,
    NVPTX,
    AMDGPU,
};

enum class eOptimizationLevel {
    O0 = 0,
    O1 = 1,
    O2 = 2,
    O3 = 3,
};

struct CompileOptions {
    eTargetArch targetArch;
    eOptimizationLevel optimizationLevel;
    OptimizationOptions optimizationOptions;
};


class CompiledModule {
    struct FunctionType {
        std::vector<ast::TypePtr> parameters;
        std::vector<ast::TypePtr> returns;
    };

public:
    CompiledModule(std::shared_ptr<ast::Module> ast, CompileOptions options);
    CompiledModule(dag::ModuleOp ast, CompileOptions options);

    void Compile(bool recordStages = false);
    pybind11::object Invoke(std::string function, pybind11::args args);
    std::vector<StageResult> GetStageResults() const;

private:
    static auto ExtractFunctions(std::shared_ptr<ast::Module> ast) -> std::unordered_map<std::string, FunctionType>;
    static auto ExtractFunctions(dag::ModuleOp ir) -> std::unordered_map<std::string, FunctionType>;

private:
    mlir::MLIRContext m_context;
    std::unique_ptr<Runner> m_runner;
    std::variant<dag::ModuleOp, std::shared_ptr<ast::Module>> m_ir;
    CompileOptions m_options;
    std::unordered_map<std::string, FunctionType> m_functions;
    std::vector<StageResult> m_stageResults;
};
