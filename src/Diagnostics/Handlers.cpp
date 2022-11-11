#include "Handlers.hpp"


ScopedDiagnosticCollector::ScopedDiagnosticCollector(mlir::MLIRContext& context)
    : m_handler(&context, [this](mlir::Diagnostic& diag) {
          m_diagnostics.push_back(std::move(diag));
      }) {}


std::vector<mlir::Diagnostic> ScopedDiagnosticCollector::TakeDiagnostics() {
    return std::move(m_diagnostics);
}