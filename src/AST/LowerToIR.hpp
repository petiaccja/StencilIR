#pragma once


#include "mlir/IR/MLIRContext.h"
#include <AST/AST.hpp>

#include <mlir/IR/BuiltinOps.h>


mlir::ModuleOp LowerToIR(mlir::MLIRContext& context, const ast::Module& node);