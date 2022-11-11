#pragma once

#include <mlir/IR/Diagnostics.h>

#include <stdexcept>
#include <vector>


class Exception : public std::exception {
public:
    Exception(std::string message) : m_message(std::move(message)) {}

    const char* what() const noexcept override { return m_message.c_str(); }

private:
    std::string m_message;
};


class NotImplementedError : public Exception {
public:
    using Exception::Exception;
};


class UndefinedSymbolError : public Exception {
public:
    UndefinedSymbolError(mlir::Location location, std::string symbol);

private:
    mlir::Location m_location;
    std::string m_symbol;
};


class OperandTypeError : public Exception {
public:
    OperandTypeError(mlir::Location location, std::vector<std::string> types);

    const std::vector<std::string>& Types() const { return m_types; }

private:
    std::vector<std::string> m_types;
};


class ArgumentTypeError : public Exception {
public:
    ArgumentTypeError(mlir::Location location, std::string type, int argumentIndex);
    const std::string& Type() const { return m_type; }
    int ArgumentIndex() const { return m_argumentIndex; }

private:
    std::string m_type;
    int m_argumentIndex;
};


class ArgumentCountError : public Exception {
public:
    ArgumentCountError(mlir::Location location, int expected, int provided);

    int Expected() const { return m_expected; }
    int Provided() const { return m_provided; }

private:
    int m_expected;
    int m_provided;
};


class DiagnosticError : public Exception {
public:
    DiagnosticError(std::vector<mlir::Diagnostic> diagnostics);

    const std::vector<mlir::Diagnostic>& Diagnostics() const { return m_diagnostics; }

private:
    std::vector<mlir::Diagnostic> m_diagnostics;
};


class InternalDiagnosticError : public DiagnosticError {
    using DiagnosticError::DiagnosticError;
};