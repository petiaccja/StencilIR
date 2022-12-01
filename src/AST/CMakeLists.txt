find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

add_library(AST STATIC)

target_sources(AST
    PRIVATE
        Building.hpp
        Nodes.hpp
        Types.hpp
        IRGenerator.hpp
        SymbolTable.hpp
        ConvertASTToIR.cpp
        ConvertASTToIR.hpp
)
target_include_directories(AST INTERFACE ${CMAKE_CURRENT_LIST_DIR}/..)

target_include_directories(AST PUBLIC ${LLVM_INCLUDE_DIRS})
target_include_directories(AST PUBLIC ${MLIR_INCLUDE_DIRS})
target_link_directories(AST PUBLIC ${LLVM_BUILD_LIBRARY_DIR})
target_compile_definitions(AST PUBLIC ${LLVM_DEFINITIONS})

target_link_libraries(AST StencilDialect Diagnostics)

target_link_libraries(AST
    MLIRIR
)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(AST ${dialect_libs})