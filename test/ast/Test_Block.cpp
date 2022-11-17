#include "Checker.hpp"

#include <AST/Building.hpp>

#include <catch2/catch.hpp>


TEST_CASE("Block - without return value", "[AST]") {
    const auto ast = ast::module_({
        ast::function("fun",
                      {},
                      {},
                      {
                          ast::block({
                            ast::constant(1.0f),
                            ast::yield()
                          }),
                          ast::return_(),
                      }),
    });

    const auto pattern = R"(
        // CHECK: scf.execute_region
    )";

    REQUIRE(Check(*ast, pattern));
}

TEST_CASE("Block - with return value", "[AST]") {
    const auto ast = ast::module_({
        ast::function("fun",
                      {},
                      {},
                      {
                          ast::block({
                            ast::constant(1.0f),
                            ast::yield({ast::constant(2.0f)})
                          }),
                          ast::return_(),
                      }),
    });

    const auto pattern = R"(
        // CHECK: %[[RESULT:.*]] = scf.execute_region
    )";

    REQUIRE(Check(*ast, pattern));
}