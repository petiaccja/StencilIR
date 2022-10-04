#pragma once


#include <AST/Nodes.hpp>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>


mlir::ModuleOp ConvertASTToIR(mlir::MLIRContext& context, const ast::Module& node);