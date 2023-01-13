project(ExternalLLVM)

cmake_minimum_required(VERSION 3.20)

include(ExternalProject)

set(llvm_cmake_args
    "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-G" "${CMAKE_GENERATOR}"
)

set(llvm_cache_args
    "-DLLVM_ENABLE_PROJECTS:STRING=mlir"
    "-DLLVM_TARGETS_TO_BUILD:STRING=X86\;NVPTX\;AMDGPU"
    "-DLLVM_BUILD_TOOLS:BOOL=OFF"
    "-DLLVM_ENABLE_BINDINGS:BOOL=OFF"
    "-DLLVM_ENABLE_LLD:BOOL=ON"
    "-DLLVM_ENABLE_ASSERTIONS:BOOL=ON"
    "-DLLVM_ENABLE_RTTI:BOOL=ON"
    "-DLLVM_ENABLE_EH:BOOL=ON"
    "-DLLVM_USE_PERF:BOOL=OFF"
    "-DLLVM_USE_INTEL_JITEVENTS:BOOL=OFF"
)

set(LLVM_PROJECT_PREFIX "${CMAKE_BINARY_DIR}/llvm-project")

ExternalProject_Add(llvm-project
    URL "https://github.com/llvm/llvm-project/releases/download/llvmorg-15.0.6/llvm-project-15.0.6.src.tar.xz"
    SOURCE_SUBDIR llvm
    PREFIX "${LLVM_PROJECT_PREFIX}"
    STAMP_DIR "${LLVM_PROJECT_PREFIX}/stamp"
    DOWNLOAD_DIR "${LLVM_PROJECT_PREFIX}/download"
    SOURCE_DIR "${LLVM_PROJECT_PREFIX}/src"
    BINARY_DIR "${LLVM_PROJECT_PREFIX}/build"
    INSTALL_DIR "${CMAKE_BINARY_DIR}/install"
    CMAKE_ARGS ${llvm_cmake_args} "-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/install"
    CMAKE_CACHE_ARGS ${llvm_cache_args}
    BUILD_ALWAYS OFF
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
)

add_executable(ExternalLLVM ExternalLLVM.cpp)