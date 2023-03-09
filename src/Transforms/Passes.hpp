#pragma once


#include "ReduceDimOps.hpp"


inline auto createReduceDimOpsPass() {
    return std::make_unique<ReduceDimOpsPass>();
}