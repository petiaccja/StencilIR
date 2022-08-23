#pragma once

#include <mlir/IR/DialectRegistry.h>

namespace stencil {

void registerBufferizableOpInterfaceExternalModels(mlir::DialectRegistry& registry);

}
