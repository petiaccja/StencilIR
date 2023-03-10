find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

add_library(DAG STATIC)

target_sources(DAG
    PRIVATE
        Operation.cpp
        Operation.hpp
        Converter.cpp
        Converter.hpp
        Ops.hpp
        ConvertOps.cpp
        ConvertOps.hpp
)
target_include_directories(DAG INTERFACE ${CMAKE_CURRENT_LIST_DIR}/..)

target_include_directories(DAG SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS})
target_include_directories(DAG SYSTEM PUBLIC ${MLIR_INCLUDE_DIRS})
target_link_directories(DAG PUBLIC ${LLVM_BUILD_LIBRARY_DIR})
target_compile_definitions(DAG PUBLIC ${LLVM_DEFINITIONS})

target_link_libraries(DAG StencilDialect Diagnostics AST)

target_link_libraries(DAG
    MLIRIR
)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(DAG ${dialect_libs})