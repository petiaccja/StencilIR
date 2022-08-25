add_library(AST STATIC)

target_sources(AST
    PRIVATE
        ASTBuilding.hpp
        ASTNodes.hpp
        ASTTypes.hpp
        IRGenerator.hpp
        SymbolTable.hpp
        ConvertASTToIR.cpp
        ConvertASTToIR.hpp
)
target_include_directories(AST INTERFACE ${CMAKE_CURRENT_LIST_DIR}/..)
target_link_libraries(AST StencilDialect)