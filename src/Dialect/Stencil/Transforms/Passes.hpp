#include "DeduplicateApplyInputs.hpp"
#include "FuseApplyOps.hpp"
#include "FuseExtractSliceOps.hpp"


inline auto createFuseApplyOpsPass() {
    return std::make_unique<FuseApplyOpsPass>();
}

inline auto createFuseExtractSliceOpsPass() {
    return std::make_unique<FuseExtractSliceOpsPass>();
}

inline auto createDeduplicateApplyInputsPass() {
    return std::make_unique<DeduplicateApplyInputsPass>();
}
