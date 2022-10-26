#include "Checker.hpp"

#include <AST/Building.hpp>

#include <catch2/catch.hpp>


TEST_CASE("Call", "[AST]") {
    const auto ast = ast::module_({
        ast::function("callee",
                      { ast::Parameter{ "a", ast::ScalarType::FLOAT32 } },
                      { ast::ScalarType::FLOAT32 },
                      { ast::return_({ ast::symref("a") }) }),
        ast::function("caller",
                      {},
                      {},
                      {
                          ast::call("callee", { ast::constant(1.0f) }),
                          ast::return_(),
                      }),
    });

    const auto pattern = R"(
        // CHECK: %[[RESULT:.*]] = call @callee(%[[ARG:.*]])
    )";

    REQUIRE(Check(*ast, pattern));
}