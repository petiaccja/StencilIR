#include "Exception.hpp"

#include "Formatting.hpp"

#include <sstream>

namespace sir {

static std::string FormatDiagVector(const std::vector<mlir::Diagnostic>& diagnostics) {
    std::stringstream ss;
    for (const auto& diag : diagnostics) {
        ss << FormatDiagnostic(diag) << std::endl;
    }
    return ss.str();
}

CompilationError::CompilationError(const std::vector<mlir::Diagnostic>& diagnostics, mlir::ModuleOp moduleOp)
    : SyntaxError(FormatDiagVector(diagnostics)),
      m_moduleOp(FormatModule(moduleOp)) {}


CompilationError::CompilationError(const std::vector<mlir::Diagnostic>& diagnostics)
    : SyntaxError(FormatDiagVector(diagnostics)) {}


UndefinedSymbolError::UndefinedSymbolError(mlir::Location location, std::string symbol)
    : SyntaxError([&]() {
          return FormatDiagnostic(FormatLocation(location),
                                  FormatSeverity(mlir::DiagnosticSeverity::Error),
                                  "undefined symbol: " + symbol);
      }()) {}


OperandTypeError::OperandTypeError(mlir::Location location, std::vector<std::string> types)
    : SyntaxError([&]() {
          std::stringstream message;
          message << "operand types incompatible: ";
          for (const auto& type : types) {
              message << type;
              if (!types.empty() && &type != &types.back()) {
                  message << ", ";
              }
          }
          return FormatDiagnostic(FormatLocation(location),
                                  FormatSeverity(mlir::DiagnosticSeverity::Error),
                                  message.str());
      }()) {}


ArgumentTypeError::ArgumentTypeError(mlir::Location location, std::string type, int argumentIndex)
    : SyntaxError([&]() {
          std::stringstream message;
          message << "argument " << argumentIndex << " has incompatible type: " << type;
          return FormatDiagnostic(FormatLocation(location),
                                  FormatSeverity(mlir::DiagnosticSeverity::Error),
                                  message.str());
      }()) {}


ArgumentCountError::ArgumentCountError(mlir::Location location, int expected, int provided)
    : SyntaxError([&]() {
          std::stringstream message;
          message << "expected " << expected << " arguments but " << provided << " was provided";
          return FormatDiagnostic(FormatLocation(location),
                                  FormatSeverity(mlir::DiagnosticSeverity::Error),
                                  message.str());
      }()) {}


} // namespace sir