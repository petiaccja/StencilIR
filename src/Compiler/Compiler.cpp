#include "Compiler.hpp"
#include <Diagnostics/Handlers.hpp>
#include <Diagnostics/Exception.hpp>


mlir::ModuleOp Compiler::Run(mlir::ModuleOp module) const {
    std::vector<StageResult> stageResults;
    return Run(module, stageResults, false);
}

mlir::ModuleOp Compiler::Run(mlir::ModuleOp module, std::vector<StageResult>& stageResults) const {
    return Run(module, stageResults, true);
}

static std::string to_string(mlir::ModuleOp module) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    module.print(ss);
    return s;
}

mlir::ModuleOp Compiler::Run(mlir::ModuleOp module, std::vector<StageResult>& stageResults, bool printStageResults) const {
    // Clone because we don't want to modify the original.
    module = module.clone();
    auto& context = *module->getContext();

    ScopedDiagnosticCollector diagnostics{context};

    size_t index = 0;
    stageResults.push_back({ std::to_string(index++) + "_input", to_string(module) });

    for (const auto& stage : m_stages) {
        const auto passesResult = stage.passes->run(module);
        if (failed(passesResult)) {
            if (printStageResults) {
                stageResults.push_back({ std::to_string(index) + "_" + stage.name, to_string(module) });
            }
            mlir::Diagnostic stageNote{mlir::UnknownLoc::get(&context), mlir::DiagnosticSeverity::Remark};
            stageNote << "ICE occured in stage \"" << stage.name << "\"";
            auto diagList = diagnostics.TakeDiagnostics();
            diagList.push_back(std::move(stageNote));
            throw InternalDiagnosticError(std::move(diagList));
        }
        if (printStageResults) {
            stageResults.push_back({ std::to_string(index) + "_" + stage.name, to_string(module) });
        }
        ++index;
    }

    return module;
}