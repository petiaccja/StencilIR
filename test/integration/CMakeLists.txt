add_executable(test_integration)

target_sources(test_integration
    PRIVATE
        main.cpp
        Utility/RunAST.hpp
)

target_sources(test_integration
    PRIVATE
        Structured.cpp
        Unstructured.cpp
)

target_link_libraries(test_integration AST Compiler Execution)

find_package(Catch2)
target_link_libraries(test_integration Catch2::Catch2)