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
add_library(MockDialect STATIC)

target_sources(MockDialect
    PRIVATE
        MockDialect.cpp
        MockOps.cpp
)

target_include_directories(MockDialect PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/include)
target_include_directories(MockDialect INTERFACE ${CMAKE_CURRENT_LIST_DIR}/..)

target_include_directories(MockDialect PUBLIC ${LLVM_INCLUDE_DIRS})
target_include_directories(MockDialect PUBLIC ${MLIR_INCLUDE_DIRS})
target_link_directories(MockDialect PUBLIC ${LLVM_BUILD_LIBRARY_DIR})
target_compile_definitions(MockDialect PUBLIC ${LLVM_DEFINITIONS})

target_link_libraries(MockDialect
    MLIRIR
    MLIRInferTypeOpInterface
    MLIRCallInterfaces
    MLIRSideEffectInterfaces
    MLIRCastInterfaces
)

# Dependencies
add_dependencies(MockDialect mlir-headers)
