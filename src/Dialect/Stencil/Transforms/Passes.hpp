#include "FuseApplyOps.hpp"


inline auto createFuseApplyOpsPass() {
    return std::make_unique<FuseApplyOpsPass>();
}
