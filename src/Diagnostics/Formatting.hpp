#pragma once


#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>

#include <optional>
#include <string>



std::string IndentText(std::string_view input, int spaces);
std::optional<std::string> FormatSeverity(mlir::DiagnosticSeverity severity);
std::string FormatLocation(std::string_view file, int line, int column);
std::optional<std::string> FormatLocation(mlir::Location location);
std::string FormatDiagnostic(std::optional<std::string> location,
                             std::optional<std::string> severity,
                             std::string message);
std::string FormatDiagnostic(const mlir::Diagnostic& diag);
std::string FormatModule(mlir::ModuleOp& moduleOp);