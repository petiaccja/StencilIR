#include <AST/Building.hpp>
#include <AST/Nodes.hpp>



template <class... Statement>
std::shared_ptr<ast::Module> EncloseStatements(Statement... statements) {
    return ast::module_({
        ast::function("fun",
                      {},
                      {},
                      {
                          statements...,
                          ast::return_(),
                      }),
    });
}