#pragma once

#include <mlir/Dialect/Bufferization/Transforms/Passes.h>


std::unique_ptr<mlir::Pass> createOneShotBufferizeCombinedPass(const mlir::bufferization::OneShotBufferizationOptions& options);