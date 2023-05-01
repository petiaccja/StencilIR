#pragma once


namespace mlir {
class DialectRegistry;
}


namespace stencil {
void registerRuntimeVerifiableOpInterfaceExternalModels(mlir::DialectRegistry& registry);
} // namespace stencil
