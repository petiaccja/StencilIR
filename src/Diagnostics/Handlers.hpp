#pragma once

#include <mlir/IR/Diagnostics.h>


namespace sir {


class ScopedDiagnosticCollector {
public:
    ScopedDiagnosticCollector(mlir::MLIRContext& context);
    ScopedDiagnosticCollector(ScopedDiagnosticCollector&&) = delete;
    ScopedDiagnosticCollector& operator=(ScopedDiagnosticCollector&&) = delete;

    std::vector<mlir::Diagnostic> TakeDiagnostics();

private:
    mlir::ScopedDiagnosticHandler m_handler;
    std::vector<mlir::Diagnostic> m_diagnostics;
};


} // namespace sir