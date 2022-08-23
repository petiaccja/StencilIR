#pragma once

#include "PrintToLLVMPass.hpp"
#include "StencilToStd.hpp"


inline auto createStencilToLoopFuncPass() {
    auto pass = std::make_unique<StencilToLoopFuncPass>();
    return pass;
}

inline auto createStencilToStdPass() {
    auto pass = std::make_unique<StencilToStdPass>();
    return pass;
}

inline auto createStencilPrintToLLVMPass() {
    return std::make_unique<PrintToLLVMPass>();
}