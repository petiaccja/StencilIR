include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(MLIR_TABLEGEN_EXE "${MLIR_INSTALL_PREFIX}/bin/${MLIR_TABLEGEN_EXE}")

add_mlir_dialect(Stencil stencil)