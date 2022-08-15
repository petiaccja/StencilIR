#pragma once

#include "PrintToLLVMPass.hpp"
#include "StencilToStd.hpp"


inline auto createStencilToStdPass(bool launchToGpu = false) {
    auto pass = std::make_unique<StencilToStdPass>();
    pass->launchToGpu = launchToGpu;
    return pass;
}

inline auto createStencilPrintToLLVMPass() {
    return std::make_unique<PrintToLLVMPass>();
}