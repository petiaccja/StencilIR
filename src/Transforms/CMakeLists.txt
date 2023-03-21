find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

add_library(Transforms)

target_sources(Transforms
    PRIVATE
        ReduceDimOps.cpp
        ReduceDimOps.hpp
        EliminateUnusedAllocTensors.cpp
        EliminateUnusedAllocTensors.hpp
)

target_include_directories(Transforms INTERFACE ${CMAKE_CURRENT_LIST_DIR}/..)

target_include_directories(Transforms SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS})
target_include_directories(Transforms SYSTEM PUBLIC ${MLIR_INCLUDE_DIRS})
target_link_directories(Transforms PUBLIC ${LLVM_BUILD_LIBRARY_DIR})
target_compile_definitions(Transforms PUBLIC ${LLVM_DEFINITIONS})

target_link_libraries(Transforms StencilDialect AST Diagnostics)

target_link_libraries(Transforms
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRTransforms
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(Transforms ${dialect_libs})