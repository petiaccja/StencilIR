add_executable(TestStructured main.cpp)

target_link_libraries(TestStructured Execution StencilDialect AST Compiler)
