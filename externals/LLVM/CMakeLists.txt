project(ExternalLLVM)

cmake_minimum_required(VERSION 3.20)

include(ExternalProject)

set(options
    "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-G" ${CMAKE_GENERATOR}
    "-DLLVM_ENABLE_PROJECTS=mlir"
    "-DLLVM_TARGETS_TO_BUILD=X86\;NVPTX\;AMDGPU"
    "-DLLVM_BUILD_TOOLS=OFF"
    "-DLLVM_ENABLE_BINDINGS=OFF"
    "-DLLVM_ENABLE_LLD=ON"
    "-DLLVM_ENABLE_ASSERTIONS=ON"
    "-DLLVM_ENABLE_RTTI=ON"
    "-DLLVM_ENABLE_EH=ON"
    "-DLLVM_USE_PERF=OFF"
    "-DLLVM_USE_INTEL_JITEVENTS=OFF"
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
    CMAKE_ARGS ${options} "-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/install"
    BUILD_ALWAYS OFF
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
)

add_executable(ExternalLLVM ExternalLLVM.cpp)