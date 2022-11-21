#pragma once

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>

#include <stdexcept>
#include <vector>


//------------------------------------------------------------------------------
// Base class
//------------------------------------------------------------------------------
class Exception : public std::exception {
public:
    Exception(std::string message) : m_message(std::move(message)) {}

    const char* what() const noexcept override { return m_message.c_str(); }
    std::string_view GetMessage() const noexcept { return m_message; }

private:
    std::string m_message;
};


//------------------------------------------------------------------------------
// Special errors
//------------------------------------------------------------------------------
class NotImplementedError : public Exception {
public:
    using Exception::Exception;
};


//------------------------------------------------------------------------------
// Syntax errors
//------------------------------------------------------------------------------
class SyntaxError : public Exception {
public:
    using Exception::Exception;
};


class CompilationError : public SyntaxError {
public:
    CompilationError(const std::vector<mlir::Diagnostic>& diagnostics);
    CompilationError(const std::vector<mlir::Diagnostic>& diagnostics, mlir::ModuleOp module);

    std::string_view GetModule() const noexcept { return m_module; }

private:
    std::string m_module;
};


class UndefinedSymbolError : public SyntaxError {
public:
    UndefinedSymbolError(mlir::Location location, std::string symbol);
};


class OperandTypeError : public SyntaxError {
public:
    OperandTypeError(mlir::Location location, std::vector<std::string> types);
};


class ArgumentTypeError : public SyntaxError {
public:
    ArgumentTypeError(mlir::Location location, std::string type, int argumentIndex);
};


class ArgumentCountError : public SyntaxError {
public:
    ArgumentCountError(mlir::Location location, int expected, int provided);
};