#include "Checker.hpp"

#include <AST/Building.hpp>

#include <catch2/catch.hpp>


TEST_CASE("Min", "[AST]") {
    const auto ast = ast::module_({
        ast::function("fun",
                      {},
                      {},
                      {
                          ast::min(ast::constant(1.0f), ast::constant(2.0f)),
                          ast::return_(),
                      }),
    });

    const auto pattern = R"(
        // CHECK: minf
    )";

    REQUIRE(Check(*ast, pattern));
}

TEST_CASE("Max", "[AST]") {
    const auto ast = ast::module_({
        ast::function("fun",
                      {},
                      {},
                      {
                          ast::max(ast::constant(1), ast::constant(2)),
                          ast::return_(),
                      }),
    });

    const auto pattern = R"(
        // CHECK: maxsi
    )";

    REQUIRE(Check(*ast, pattern));
}