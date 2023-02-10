#include "FuseApplyOps.hpp"
#include "FuseExtractSliceOps.hpp"


inline auto createFuseApplyOpsPass() {
    return std::make_unique<FuseApplyOpsPass>();
}

inline auto createFuseExtractSliceOps() {
    return std::make_unique<FuseExtractSliceOpsPass>();
}
