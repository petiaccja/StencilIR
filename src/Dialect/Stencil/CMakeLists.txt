find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Tablegen target
add_subdirectory(include)

# Library target
add_library(StencilDialect STATIC)

target_sources(StencilDialect
    PRIVATE
        IR/StencilOps.cpp
        IR/StencilOps.hpp
        Transforms/BufferizableOpInterfaceImpl.cpp
        Transforms/BufferizableOpInterfaceImpl.hpp
        Transforms/FuseApplyOps.cpp
        Transforms/FuseApplyOps.hpp
        Transforms/FuseExtractSliceOps.cpp
        Transforms/FuseExtractSliceOps.hpp
        Transforms/DeduplicateApplyInputs.cpp
        Transforms/DeduplicateApplyInputs.hpp
        Transforms/Utility.cpp
        Transforms/Utility.hpp
        Transforms/RuntimeVerificationInterfaceImpl.cpp
        Transforms/RuntimeVerificationInterfaceImpl.hpp
)

target_include_directories(StencilDialect PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/include)
target_include_directories(StencilDialect PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../..)

target_include_directories(StencilDialect SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS})
target_include_directories(StencilDialect SYSTEM PUBLIC ${MLIR_INCLUDE_DIRS})
target_link_directories(StencilDialect PUBLIC ${LLVM_BUILD_LIBRARY_DIR})
target_compile_definitions(StencilDialect PUBLIC ${LLVM_DEFINITIONS})

target_link_libraries(StencilDialect
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRCallInterfaces
    MLIRSideEffectInterfaces
    MLIRCastInterfaces
)

# Dependencies
add_dependencies(StencilDialect mlir-headers)
