#include "MockOps.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypes.h>

// clang-format: off
#define GET_OP_CLASSES
#include <MockDialect/Mock.cpp.inc>
// clang-format: on