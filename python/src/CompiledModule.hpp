#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Compiler/Pipelines.hpp>
#include <Execution/Execution.hpp>
#include <IR/Ops.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>


namespace sir {


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
    CompiledModule(ops::ModuleOp ast, CompileOptions options);

    void Compile(bool recordStages = false);
    pybind11::object Invoke(std::string function, pybind11::args args);
    std::vector<StageResult> GetStageResults() const;
    std::string GetLLVMIR() const;
    std::vector<char> GetObjectFile() const;

private:
    static auto ExtractFunctions(ops::ModuleOp ir) -> std::unordered_map<std::string, FunctionType>;

private:
    mlir::MLIRContext m_context;
    std::unique_ptr<Runner> m_runner;
    ops::ModuleOp m_ir;
    CompileOptions m_options;
    std::unordered_map<std::string, FunctionType> m_functions;
    std::vector<StageResult> m_stageResults;
};


} // namespace sir