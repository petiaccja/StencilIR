#include "Formatting.hpp"

#include <mlir/IR/BuiltinAttributes.h>

#include <sstream>

using namespace std::string_literals;


std::string IndentText(std::string_view input, int spaces) {
    std::string s;
    s.reserve(input.size() + spaces * 3);
    s.append(' ', spaces);
    for (auto& c : input) {
        if (c == '\n') {
            s.append(' ', spaces);
        }
        s.push_back(c);
    }
    return s;
}


std::optional<std::string> FormatSeverity(mlir::DiagnosticSeverity severity) {
    switch (severity) {
        case mlir::DiagnosticSeverity::Note: return "note";
        case mlir::DiagnosticSeverity::Warning: return "warning";
        case mlir::DiagnosticSeverity::Error: return "error";
        case mlir::DiagnosticSeverity::Remark: return "remark";
    }
    return {};
}


std::string FormatLocation(std::string_view file, int line, int column) {
    std::stringstream ss;
    ss << file
       << ":"
       << line
       << ":"
       << column;
    return ss.str();
}


std::optional<std::string> FormatLocation(mlir::Location location) {
    if (auto fileLocation = location.dyn_cast<mlir::FileLineColLoc>()) {
        return FormatLocation(fileLocation.getFilename().data(), fileLocation.getLine(), fileLocation.getColumn());
    }
    return {};
}


std::string FormatDiagnostic(std::optional<std::string> location,
                             std::optional<std::string> severity,
                             std::string message) {
    std::stringstream ss;
    ss << location.value_or(""s) << (location ? ": " : "")
       << severity.value_or(""s) << (severity ? ": " : "")
       << message;
    return ss.str();
}


std::string FormatDiagnostic(const mlir::Diagnostic& diag) {
    const auto severity = FormatSeverity(diag.getSeverity());
    const auto location = FormatLocation(diag.getLocation());
    const auto message = diag.str();

    std::stringstream ss;
    ss << FormatDiagnostic(location, severity, message);
    for (auto& noteDiag : diag.getNotes()) {
        const auto note = FormatDiagnostic(noteDiag);
        ss << std::endl << IndentText(note, 4);
    }
    return ss.str();
}


std::string FormatModule(mlir::ModuleOp& module) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    module.print(ss);
    return s;
}