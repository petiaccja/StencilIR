find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

add_library(Diagnostics STATIC)

target_sources(Diagnostics
    PRIVATE
        Exception.cpp
        Exception.hpp
        Handlers.cpp
        Handlers.hpp
        Formatting.cpp
        Formatting.hpp
)

target_include_directories(Diagnostics INTERFACE ${CMAKE_CURRENT_LIST_DIR}/..)

target_include_directories(Diagnostics PUBLIC ${LLVM_INCLUDE_DIRS})
target_include_directories(Diagnostics PUBLIC ${MLIR_INCLUDE_DIRS})
target_link_directories(Diagnostics PUBLIC ${LLVM_BUILD_LIBRARY_DIR})
target_compile_definitions(Diagnostics PUBLIC ${LLVM_DEFINITIONS})

target_link_libraries(Diagnostics
    MLIRIR
    MLIRSupport
)