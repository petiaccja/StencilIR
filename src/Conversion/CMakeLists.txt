find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

add_library(Conversion)

target_sources(Conversion
    PRIVATE
        StencilPrintToLLVM/StencilPrintToLLVM.cpp
        StencilPrintToLLVM/StencilPrintToLLVM.hpp
        StencilApplyToLoops/StencilToLoops.cpp
        StencilApplyToLoops/StencilToLoops.hpp
        StencilToStandard/StencilToStandard.cpp
        StencilToStandard/StencilToStandard.hpp
        StencilToFunc/StencilToFunc.cpp
        StencilToFunc/StencilToFunc.hpp
)

target_include_directories(Conversion INTERFACE ${CMAKE_CURRENT_LIST_DIR}/..)

target_include_directories(Conversion SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS})
target_include_directories(Conversion SYSTEM PUBLIC ${MLIR_INCLUDE_DIRS})
target_link_directories(Conversion PUBLIC ${LLVM_BUILD_LIBRARY_DIR})
target_compile_definitions(Conversion PUBLIC ${LLVM_DEFINITIONS})

target_link_libraries(Conversion StencilDialect Diagnostics)

target_link_libraries(Conversion
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRTransforms
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(Conversion ${dialect_libs})