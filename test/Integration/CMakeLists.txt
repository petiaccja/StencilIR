add_executable(TestIntegration)

target_sources(TestIntegration
    PRIVATE
        main.cpp
        Utility/RunAST.hpp
)

target_sources(TestIntegration
    PRIVATE
        Test_Structured.cpp
        Test_Unstructured.cpp
)

target_link_libraries(TestIntegration AST Compiler Execution)

find_package(Catch2)
target_link_libraries(TestIntegration Catch2::Catch2)