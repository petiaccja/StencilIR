find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Library target
add_library(BufferizationExtensions STATIC)

target_sources(BufferizationExtensions
    PRIVATE
        Transforms/OneShotBufferizeCombined.cpp
        Transforms/OneShotBufferizeCombined.hpp
)

target_include_directories(BufferizationExtensions PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/include)
target_include_directories(BufferizationExtensions INTERFACE ${CMAKE_CURRENT_LIST_DIR}/../..)

target_include_directories(BufferizationExtensions SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS})
target_include_directories(BufferizationExtensions SYSTEM PUBLIC ${MLIR_INCLUDE_DIRS})
target_link_directories(BufferizationExtensions PUBLIC ${LLVM_BUILD_LIBRARY_DIR})
target_compile_definitions(BufferizationExtensions PUBLIC ${LLVM_DEFINITIONS})

target_link_libraries(BufferizationExtensions
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRCallInterfaces
    MLIRSideEffectInterfaces
    MLIRCastInterfaces
)

# Dependencies
add_dependencies(BufferizationExtensions mlir-headers)
