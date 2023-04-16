#pragma once


#include "EliminateUnusedAllocTensors.hpp"
#include "ReduceDimOps.hpp"


namespace sir {

inline auto createReduceDimOpsPass() {
    return std::make_unique<ReduceDimOpsPass>();
}

inline auto createEliminateUnusedAllocTensorsPass() {
    return std::make_unique<EliminateUnusedAllocTensorsPass>();
}


} // namespace sir