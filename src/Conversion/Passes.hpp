#pragma once

#include "StencilPrintToLLVM/StencilPrintToLLVM.hpp"
#include "StencilApplyToLoops/StencilApplyToLoops.hpp"
#include "StencilOpsToStandard/StencilOpsToStandard.hpp"


inline auto createStencilApplyToLoopsPass() {
    auto pass = std::make_unique<StencilApplyToLoopsPass>();
    return pass;
}

inline auto createStencilOpsToStandardPass() {
    auto pass = std::make_unique<StencilOpsToStandardPass>();
    return pass;
}

inline auto createStencilPrintToLLVMPass() {
    return std::make_unique<StencilPrintToLLVMPass>();
}