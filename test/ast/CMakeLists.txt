find_package(MLIR REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

add_executable(TestAST)

target_sources(TestAST
    PRIVATE
        main.cpp
        Checker.hpp
        Checker.cpp
)

target_sources(TestAST
    PRIVATE
        Test_Function.cpp
        Test_Call.cpp
        Test_If.cpp
        Test_Return.cpp
        Test_Yield.cpp
        Test_Assign.cpp
)

target_link_libraries(TestAST LLVMFileCheck LLVMSupport)
target_link_libraries(TestAST AST)

find_package(Catch2)
target_link_libraries(TestAST Catch2::Catch2)