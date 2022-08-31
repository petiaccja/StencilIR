find_package(pybind11 REQUIRED)

pybind11_add_module(stencilir_ext)

target_sources(stencilir_ext
    PRIVATE
        StencilIRExt.cpp)

target_link_libraries(stencilir_ext PUBLIC AST Compiler Execution)