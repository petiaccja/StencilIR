find_package(pybind11 REQUIRED)

pybind11_add_module(stencilir)

target_sources(stencilir
    PRIVATE
        Bindings.cpp)

target_link_libraries(stencilir PUBLIC AST Compiler Execution)