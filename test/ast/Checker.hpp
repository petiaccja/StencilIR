#pragma once

#include <AST/Nodes.hpp>

#include <mlir/IR/BuiltinOps.h>

#include <string_view>


std::string PrintIr(mlir::ModuleOp module);
bool Check(std::string_view input, std::string_view pattern);
bool Check(ast::Module& module, std::string_view pattern);