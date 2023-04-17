#include "Compiler.hpp"

#include <Diagnostics/Exception.hpp>
#include <Diagnostics/Handlers.hpp>


namespace sir {

mlir::ModuleOp Compiler::Run(mlir::ModuleOp moduleOp) const {
    std::vector<StageResult> stageResults;
    return Run(moduleOp, stageResults, false);
}

mlir::ModuleOp Compiler::Run(mlir::ModuleOp moduleOp, std::vector<StageResult>& stageResults) const {
    return Run(moduleOp, stageResults, true);
}

static std::string to_string(mlir::ModuleOp moduleOp) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    moduleOp.print(ss);
    return s;
}

mlir::ModuleOp Compiler::Run(mlir::ModuleOp moduleOp, std::vector<StageResult>& stageResults, bool printStageResults) const {
    // Clone because we don't want to modify the original.
    moduleOp = moduleOp.clone();
    auto& context = *moduleOp->getContext();

    ScopedDiagnosticCollector diagnostics{ context };

    size_t index = 0;
    stageResults.push_back({ std::to_string(index++) + "_input", to_string(moduleOp) });

    for (const auto& stage : m_stages) {
        const auto passesResult = stage.passes->run(moduleOp);
        if (failed(passesResult)) {
            if (printStageResults) {
                stageResults.push_back({ std::to_string(index) + "_" + stage.name, to_string(moduleOp) });
            }
            mlir::Diagnostic stageNote{ mlir::UnknownLoc::get(&context), mlir::DiagnosticSeverity::Remark };
            stageNote << "ICE occured in stage \"" << stage.name << "\"";
            auto diagList = diagnostics.TakeDiagnostics();
            diagList.push_back(std::move(stageNote));
            throw CompilationError(diagList, moduleOp);
        }
        if (printStageResults) {
            stageResults.push_back({ std::to_string(index) + "_" + stage.name, to_string(moduleOp) });
        }
        ++index;
    }

    return moduleOp;
}


} // namespace sir