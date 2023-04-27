#pragma once

#include "Compiler.hpp"


namespace sir {

struct OptimizationOptions {
    bool inlineFunctions = false;
    bool fuseExtractSliceOps = false;
    bool fuseApplyOps = false;
    bool eliminateAllocBuffers = false;
    bool enableRuntimeVerification = true;
};

std::vector<Stage> TargetCPUPipeline(mlir::MLIRContext& context,
                                     const OptimizationOptions& macroOptimizationOptions = {});

} // namespace sir


namespace std {

template <>
struct hash<sir::OptimizationOptions> {
    auto operator()(const sir::OptimizationOptions& obj) const noexcept {
        std::hash<bool> h;
        auto v = h(obj.inlineFunctions);
        constexpr auto c = static_cast<decltype(v)>(8934546291568956629LL);
        v = (v * c) + h(obj.fuseExtractSliceOps);
        v = (v * c) + h(obj.fuseApplyOps);
        v = (v * c) + h(obj.eliminateAllocBuffers);
        v = (v * c) + h(obj.enableRuntimeVerification);
        return v;
    }
};

} // namespace std