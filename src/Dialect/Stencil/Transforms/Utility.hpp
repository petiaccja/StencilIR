#pragma once

#include <Dialect/Stencil/IR/StencilOps.hpp>

#include <mlir/IR/PatternMatch.h>


mlir::StringAttr UniqueStencilName(stencil::StencilOp originalStencil, mlir::PatternRewriter& rewriter);