#include "Checker.hpp"

#include <AST/Building.hpp>

#include <catch2/catch.hpp>


TEST_CASE("Return - no value", "[AST]") {
    const auto ast = ast::module_({
        ast::function("fun",
                      {},
                      {},
                      {
                          ast::return_(),
                      }),
    });

    const auto pattern = R"(
        // CHECK: return
    )";

    REQUIRE(Check(*ast, pattern));
}

TEST_CASE("Return - single value", "[AST]") {
    const auto ast = ast::module_({
        ast::function("fun",
                      {},
                      { ast::ScalarType::FLOAT32 },
                      {
                          ast::return_({ ast::constant(1.0f) }),
                      }),
    });

    const auto pattern = R"(
        // CHECK: return %[[V1:.*]] : f32
    )";

    REQUIRE(Check(*ast, pattern));
}

TEST_CASE("Return - expression unpacking", "[AST]") {
    const auto ast = ast::module_({
        ast::function("mrv",
                      {},
                      { ast::ScalarType::FLOAT32, ast::ScalarType::FLOAT32 },
                      { ast::return_({ ast::constant(1.0f), ast::constant(1.0f) }) }),
        ast::function("fun",
                      {},
                      { ast::ScalarType::FLOAT32, ast::ScalarType::FLOAT32, ast::ScalarType::SINT32 },
                      {
                          ast::return_({
                              ast::call("mrv", {}),
                              ast::constant(int32_t(1)),
                          }),
                      }),
    });

    const auto pattern = R"(
        // CHECK: return %[[F1:.*]], %[[F2:.*]], %[[I1:.*]] : f32, f32, i32
    )";

    REQUIRE(Check(*ast, pattern));
}