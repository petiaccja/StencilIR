#include "MockDialect.hpp"

#include "MockOps.hpp"
// clang-format: off
#include <MockDialect/MockDialect.cpp.inc>
// clang-format: on

void mock::MockDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include <MockDialect/Mock.cpp.inc>
        >();
}