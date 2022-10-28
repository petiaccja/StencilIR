#include "Compiler.hpp"


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

    std::stringstream diagnostics;
    mlir::ScopedDiagnosticHandler diagHandler(module->getContext(), [&](mlir::Diagnostic& diag) {
        std::string out;
        llvm::raw_string_ostream os(out);
        diag.getLocation().print(os);
        out += ": ";
        diag.print(os);
        diagnostics << "\n"
                    << out;
    });

    size_t index = 1;
    for (const auto& stage : m_stages) {
        const auto passesResult = stage.passes->run(module);
        if (failed(passesResult)) {
            if (printStageResults) {
                stageResults.push_back({ std::to_string(index) + "_" + stage.name, to_string(module) });
            }
            diagnostics << "\n"
                        << "Compilation failed during stage "
                        << "\"" << stage.name << "\"";
            throw std::runtime_error(diagnostics.str());
        }
        if (printStageResults) {
            stageResults.push_back({ std::to_string(index) + "_" + stage.name, to_string(module) });
        }
        ++index;
    }

    return module;
}