#include <AST/Building.hpp>
#include <AST/Nodes.hpp>



template <class... Statement>
std::shared_ptr<ast::Module> EncloseInFunction(Statement... statements) {
    return ast::module_(
        {
            ast::function("fun",
                          {},
                          {},
                          {
                              statements...,
                              ast::return_(),
                          }),
        });
}


template <int Dim, class... Statement>
std::shared_ptr<ast::Module> EncloseInStencil(Statement... statements) {
    return ast::module_(
        {},
        {
            ast::stencil("stn",
                         {},
                         {},
                         {
                             statements...,
                             ast::return_(),
                         },
                         Dim),
        });
}