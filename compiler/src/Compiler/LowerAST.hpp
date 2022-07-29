#pragma once


#include <AST/Node.hpp>

#include <mlir/IR/BuiltinOps.h>


mlir::ModuleOp LowerAST(ast::Module& node);