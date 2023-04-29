#pragma once

#include "StencilApplyToLoops/StencilToLoops.hpp"
#include "StencilPrintToLLVM/StencilPrintToLLVM.hpp"
#include "StencilToFunc/StencilToFunc.hpp"
#include "StencilToStandard/StencilToStandard.hpp"


namespace sir {


inline auto createStencilToLoopsPass() {
    return std::make_unique<StencilToLoopsPass>();
}

inline auto createStencilToStandardPass() {
    return std::make_unique<StencilToStandardPass>();
}

inline auto createStencilToFuncPass() {
    return std::make_unique<StencilToFuncPass>();
}

inline auto createStencilPrintToLLVMPass() {
    return std::make_unique<StencilPrintToLLVMPass>();
}


} // namespace sir