#pragma once

#include "Compiler.hpp"


struct MacroOptimizationOptions {
    bool eliminateAllocBuffers = true;
};

std::vector<Stage> TargetCPUPipeline(mlir::MLIRContext& context,
                                     const MacroOptimizationOptions& macroOptimizationOptions = {});