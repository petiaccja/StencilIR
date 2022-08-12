add_library(AST STATIC)

target_sources(AST
    PRIVATE
        Node.hpp
        AST.hpp
        Types.hpp
)
set_target_properties(AST PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(AST INTERFACE ${CMAKE_CURRENT_LIST_DIR}/..)