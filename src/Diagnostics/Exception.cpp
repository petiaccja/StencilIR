#include "Exception.hpp"

#include "Formatting.hpp"

#include <sstream>


UndefinedSymbolError::UndefinedSymbolError(mlir::Location location, std::string symbol)
    : Exception([&]() {
          return FormatDiagnostic(FormatLocation(location),
                                  FormatSeverity(mlir::DiagnosticSeverity::Error),
                                  "undefined symbol: " + symbol);
      }()),
      m_location(std::move(location)),
      m_symbol(std::move(symbol)) {}


OperandTypeError::OperandTypeError(mlir::Location location, std::vector<std::string> types)
    : Exception([&]() {
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
      }()),
      m_types(std::move(types)) {}


ArgumentTypeError::ArgumentTypeError(mlir::Location location, std::string type, int argumentIndex)
    : Exception([&]() {
          std::stringstream message;
          message << "argument " << argumentIndex << " has incompatible type: " << type;
          return FormatDiagnostic(FormatLocation(location),
                                  FormatSeverity(mlir::DiagnosticSeverity::Error),
                                  message.str());
      }()),
      m_type(std::move(type)),
      m_argumentIndex(argumentIndex) {}


ArgumentCountError::ArgumentCountError(mlir::Location location, int expected, int provided)
    : Exception([&]() {
          std::stringstream message;
          message << "expected " << expected << " arguments but " << provided << " was provided";
          return FormatDiagnostic(FormatLocation(location),
                                  FormatSeverity(mlir::DiagnosticSeverity::Error),
                                  message.str());
      }()),
      m_expected(expected),
      m_provided(provided) {}


DiagnosticError::DiagnosticError(std::vector<mlir::Diagnostic> diagnostics)
    : Exception([](auto& diags) {
          std::stringstream ss;
          for (auto& diag : diags) {
              ss << FormatDiagnostic(diag) << std::endl;
          }
          return ss.str();
      }(diagnostics)),
      m_diagnostics(std::move(diagnostics)) {}
