find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

add_library(Compiler)

target_sources(Compiler
    PRIVATE
        PrintToLLVMPass.hpp
        PrintToLLVMPass.cpp
        LoweringPasses.hpp
        LoweringPasses.cpp
        StencilLoweringPasses.cpp
        StencilLoweringPasses.hpp
        Lowering.cpp
        Lowering.hpp
)

target_include_directories(Compiler INTERFACE ${CMAKE_CURRENT_LIST_DIR}/..)

target_include_directories(Compiler PUBLIC ${LLVM_INCLUDE_DIRS})
target_include_directories(Compiler PUBLIC ${MLIR_INCLUDE_DIRS})
target_link_directories(Compiler PUBLIC ${LLVM_BUILD_LIBRARY_DIR})
target_compile_definitions(Compiler PUBLIC ${LLVM_DEFINITIONS})

target_link_libraries(Compiler StencilDialect AST)

target_link_libraries(Compiler
    MLIRAnalysis
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRTransforms
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(Compiler
    MLIRSCFDialect
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRTensorDialect
    MLIRLinalgDialect
    MLIRArithmeticDialect
    MLIRAffineDialect
    MLIRLLVMDialect
    MLIRLLVMIRTransforms
    MLIRLLVMToLLVMIRTranslation
)

target_link_libraries(Compiler
    MLIRSCFToControlFlow
    MLIRControlFlowToLLVM
    MLIRFuncToLLVM
    MLIRTensorToLinalg
    MLIRLinalgToLLVM
    MLIRArithmeticToLLVM
    MLIRAffineToStandard
    MLIRAffineTransforms
    MLIRLinalgToStandard

    MLIRGPUToVulkanTransforms
    MLIRGPUTransforms
)