#pragma once

#include "Operation.hpp"

#include <mlir/IR/Operation.h>

namespace sir {

mlir::Operation* ConvertOperation(mlir::MLIRContext& context, Operation op);

}