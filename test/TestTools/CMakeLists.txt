find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

add_library(TestTools STATIC)

target_sources(TestTools
    PRIVATE
        FileCheck.hpp
        FileCheck.cpp
)

target_include_directories(TestTools INTERFACE "${CMAKE_CURRENT_LIST_DIR}/..")

target_link_libraries(TestTools LLVMFileCheck LLVMSupport)
target_link_libraries(TestTools AST DAG)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(TestTools ${dialect_libs})