#pragma once


#include "mlir/IR/MLIRContext.h"
#include <AST/AST.hpp>

#include <mlir/IR/BuiltinOps.h>


mlir::ModuleOp LowerAST(mlir::MLIRContext& context, ast::Module& node);