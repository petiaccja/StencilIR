#include "StencilDialect.hpp"

#include "StencilOps.hpp"
// clang-format: off
#include <StencilDialect/StencilDialect.cpp.inc>
// clang-format: on

void stencil::StencilDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <StencilDialect/Stencil.cpp.inc>
        >();
}