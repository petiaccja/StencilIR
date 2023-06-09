find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

add_library(Execution STATIC)

target_sources(Execution
    PRIVATE
        Execution.cpp
        Execution.hpp
        DynamicLinking.cpp
        DynamicLinking.hpp
)

target_include_directories(Execution INTERFACE ${CMAKE_CURRENT_LIST_DIR}/..)

target_include_directories(Execution SYSTEM PUBLIC ${LLVM_INCLUDE_DIRS})
target_include_directories(Execution SYSTEM PUBLIC ${MLIR_INCLUDE_DIRS})
target_link_directories(Execution PUBLIC ${LLVM_BUILD_LIBRARY_DIR})
target_compile_definitions(Execution PUBLIC ${LLVM_DEFINITIONS})

target_link_libraries(Execution Diagnostics)

target_link_libraries(Execution
    MLIRIR
    MLIRParser
    MLIRSupport
    MLIRTransforms
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(Execution
    MLIRLLVMDialect
    MLIRLLVMIRTransforms
    MLIRLLVMToLLVMIRTranslation
    MLIRExecutionEngine
    MLIRExecutionEngineUtils
)

target_link_libraries(Execution mlir_c_runner_utils)

if (${STENCILIR_ENABLE_CUDA})
    target_link_libraries(Execution mlir_cuda_runtime)
endif()