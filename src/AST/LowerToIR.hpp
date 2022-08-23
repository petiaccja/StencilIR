#pragma once


#include <AST/ASTNodes.hpp>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>


mlir::ModuleOp LowerToIR(mlir::MLIRContext& context, const ast::Module& node);