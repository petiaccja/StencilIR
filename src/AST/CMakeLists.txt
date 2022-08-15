add_library(AST STATIC)

target_sources(AST
    PRIVATE
        Node.hpp
        AST.hpp
        Types.hpp
        LowerToIR.cpp
        LowerToIR.hpp
)
target_include_directories(AST INTERFACE ${CMAKE_CURRENT_LIST_DIR}/..)
target_link_libraries(AST StencilDialect)