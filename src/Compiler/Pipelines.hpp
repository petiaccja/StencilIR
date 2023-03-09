#pragma once

#include "Compiler.hpp"


struct OptimizationOptions {
    bool inlineFunctions = false;
    bool eliminateAllocBuffers = false;
    bool fuseApplyOps = false;
    bool fuseExtractSliceOps = false;
};

std::vector<Stage> TargetCPUPipeline(mlir::MLIRContext& context,
                                     const OptimizationOptions& macroOptimizationOptions = {});