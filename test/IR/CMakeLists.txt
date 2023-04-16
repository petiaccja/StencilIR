add_executable(TestIR)

target_sources(TestIR
    PRIVATE
        main.cpp
        Test_Module.cpp
        Test_ArithmeticLogic.cpp
        Test_ControlFlow.cpp
        Test_Stencil.cpp
        Test_Tensor.cpp
)

target_link_libraries(TestIR LLVMFileCheck LLVMSupport)
target_link_libraries(TestIR IR)
target_link_libraries(TestIR TestTools)

find_package(Catch2)
target_link_libraries(TestIR Catch2::Catch2)