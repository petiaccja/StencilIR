add_executable(TestUnstructured main.cpp)

target_link_libraries(TestUnstructured Execution StencilDialect AST Compiler)
