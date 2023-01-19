#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>

#include <memory>
#include <string>


struct Stage {
    Stage(std::string name, mlir::MLIRContext& context) : name(name),
                                                          passes(std::make_unique<mlir::PassManager>(&context)) {}
    std::string name;
    std::unique_ptr<mlir::PassManager> passes;
};

struct StageResult {
    std::string name;
    std::string ir;
};


class Compiler {
public:
    Compiler(std::vector<Stage> stages) : m_stages(std::move(stages)) {}

    mlir::ModuleOp Run(mlir::ModuleOp moduleOp) const;
    mlir::ModuleOp Run(mlir::ModuleOp moduleOp, std::vector<StageResult>& stageResults) const;

private:
    mlir::ModuleOp Run(mlir::ModuleOp moduleOp, std::vector<StageResult>& stageResults, bool printStageResults) const;

private:
    std::vector<Stage> m_stages;
};