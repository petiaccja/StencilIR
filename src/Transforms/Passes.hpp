#pragma once


#include "EliminateUnusedAllocTensors.hpp"
#include "ReduceDimOps.hpp"


inline auto createReduceDimOpsPass() {
    return std::make_unique<ReduceDimOpsPass>();
}

inline auto createEliminateUnusedAllocTensorsPass() {
    return std::make_unique<EliminateUnusedAllocTensorsPass>();
}