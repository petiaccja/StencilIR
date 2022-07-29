add_library(AST STATIC)

target_sources(AST
    PRIVATE
        Node.hpp
        Node.cpp
)

target_include_directories(AST INTERFACE ${CMAKE_CURRENT_LIST_DIR}/..)