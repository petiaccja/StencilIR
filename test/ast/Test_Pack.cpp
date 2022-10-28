#include "Checker.hpp"

#include <AST/Building.hpp>

#include <catch2/catch.hpp>


TEST_CASE("Pack - expression packing", "[AST]") {
    const auto ast = ast::module_({
        ast::function("mrv",
                      {},
                      { ast::ScalarType::FLOAT32, ast::ScalarType::FLOAT32 },
                      { ast::return_({ ast::constant(1.0f), ast::constant(1.0f) }) }),
        ast::function("fun",
                      {},
                      { ast::ScalarType::FLOAT32, ast::ScalarType::FLOAT32, ast::ScalarType::FLOAT32 },
                      {
                          ast::return_({ ast::pack({ ast::call("mrv", {}), ast::constant(1.0f) }) }),
                      }),
    });

    const auto pattern = R"(
        // CHECK: return %[[R1:.*]], %[[R2:.*]], %[[R3:.*]]
    )";

    REQUIRE(Check(*ast, pattern));
}