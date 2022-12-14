#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <AST/ConvertASTToIR.hpp>
#include <AST/Nodes.hpp>
#include <Compiler/Pipelines.hpp>
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
};


class CompiledModule {
    struct FunctionType {
        std::vector<ast::Type> parameters;
        std::vector<ast::Type> returns;
    };

public:
    CompiledModule(std::shared_ptr<ast::Module> ast, CompileOptions options, bool storeIr = false);

    pybind11::object Invoke(std::string function, pybind11::args args);
    std::vector<StageResult> GetIR() const;

private:
    static auto ExtractFunctions(std::shared_ptr<ast::Module> ast)
        -> std::unordered_map<std::string, FunctionType>;

    static Runner Compile(std::shared_ptr<ast::Module> ast,
                          CompileOptions options,
                          std::vector<StageResult>* stageResults = nullptr);

private:
    Runner m_runner;
    std::unordered_map<std::string, FunctionType> m_functions;
    std::vector<StageResult> m_ir;
};
