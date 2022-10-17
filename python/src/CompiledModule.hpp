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
public:
    CompiledModule(std::shared_ptr<ast::Module> ast, CompileOptions options, bool storeIr = false);

    void Invoke(std::string function, pybind11::args args);
    std::vector<StageResult> GetIR() const;

private:
    Runner m_runner;
    std::unordered_map<std::string, std::vector<ast::Type>> m_functions;
    std::vector<StageResult> m_ir;
};
