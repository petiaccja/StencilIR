find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

add_library(IR STATIC)

target_sources(IR
    PRIVATE
        Operation.cpp
        Operation.hpp
        Converter.cpp
        Converter.hpp
        Ops.hpp
        ConvertOps.cpp
        ConvertOps.hpp
        ConvertUtils.cpp
        ConvertUtils.hpp
        Types.cpp
        Types.hpp
)
target_include_directories(IR INTERFACE ${CMAKE_CURRENT_LIST_DIR}/..)

target_include_directories(IR SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS})
target_include_directories(IR SYSTEM PUBLIC ${MLIR_INCLUDE_DIRS})
target_link_directories(IR PUBLIC ${LLVM_BUILD_LIBRARY_DIR})
target_compile_definitions(IR PUBLIC ${LLVM_DEFINITIONS})

target_link_libraries(IR StencilDialect Diagnostics)

target_link_libraries(IR
    MLIRIR
)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(IR ${dialect_libs})