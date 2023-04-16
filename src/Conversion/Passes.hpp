#pragma once

#include "StencilApplyToLoops/StencilApplyToLoops.hpp"
#include "StencilPrintToLLVM/StencilPrintToLLVM.hpp"
#include "StencilToFunc/StencilToFunc.hpp"
#include "StencilToStandard/StencilToStandard.hpp"


namespace sir {


inline auto createStencilApplyToLoopsPass() {
    return std::make_unique<StencilApplyToLoopsPass>();
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


}