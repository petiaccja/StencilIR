#pragma once


#include "EliminateSlicing.hpp"
#include "EliminateUnusedAllocTensors.hpp"
#include "ReduceDimOps.hpp"
#include "UseCudaLibdevice.hpp"


namespace sir {

inline auto createReduceDimOpsPass() {
    return std::make_unique<ReduceDimOpsPass>();
}

inline auto createEliminateUnusedAllocTensorsPass() {
    return std::make_unique<EliminateUnusedAllocTensorsPass>();
}

inline auto createEliminateSlicingPass() {
    return std::make_unique<EliminateSlicingPass>();
}

inline auto createUseCudaLibdevicePass() {
    return std::make_unique<UseCudaLibdevicePass>();
}


} // namespace sir