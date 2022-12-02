add_executable(TestAST)

target_sources(TestAST
    PRIVATE
        main.cpp
        Test_Function.cpp
        Test_Call.cpp
        Test_If.cpp
        Test_Return.cpp
        Test_Yield.cpp
        Test_Assign.cpp
        Test_Pack.cpp
        Test_ArithmeticLogic.cpp
        Test_Block.cpp
        Test_IndexManipulation.cpp
)

target_link_libraries(TestAST LLVMFileCheck LLVMSupport)
target_link_libraries(TestAST AST)
target_link_libraries(TestAST TestTools)

find_package(Catch2)
target_link_libraries(TestAST Catch2::Catch2)