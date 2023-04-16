#pragma once

#include "Compiler.hpp"


struct OptimizationOptions {
    bool inlineFunctions = false;
    bool fuseExtractSliceOps = false;
    bool fuseApplyOps = false;
    bool eliminateAllocBuffers = false;
};

std::vector<Stage> TargetCPUPipeline(mlir::MLIRContext& context,
                                     const OptimizationOptions& macroOptimizationOptions = {});