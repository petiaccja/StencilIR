add_executable(TestGpu main.cpp)

target_link_libraries(TestGpu Execution StencilDialect AST Compiler)
