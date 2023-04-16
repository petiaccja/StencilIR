#pragma once

#include "Operation.hpp"

#include <mlir/IR/Operation.h>

namespace dag {

mlir::Operation* ConvertOperation(mlir::MLIRContext& context, Operation op);

}