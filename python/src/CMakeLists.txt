find_package(pybind11 REQUIRED)

pybind11_add_module(stencilir)

target_sources(stencilir
    PRIVATE
        Interface.cpp
        CompiledModule.cpp
        CompiledModule.hpp
        Invoke.cpp
        Invoke.hpp)

target_link_libraries(stencilir PUBLIC AST Compiler Execution Diagnostics)