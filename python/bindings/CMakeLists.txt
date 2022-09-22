find_package(pybind11 REQUIRED)

pybind11_add_module(stencilir_bindings)

target_sources(stencilir_bindings
    PRIVATE
        StencilIRBindings.cpp)

target_link_libraries(stencilir_bindings PUBLIC AST Compiler Execution)