#include "Checker.hpp"

#include <AST/Building.hpp>

#include <catch2/catch.hpp>


TEST_CASE("Yield - no value", "[AST]") {
    const auto ast = ast::module_({
        ast::function("fun",
                      {},
                      {},
                      {
                          ast::if_(ast::constant(true),
                                   { ast::yield() }),
                          ast::return_(),

                      }),
    });

    const auto pattern = R"(
        CHECK: scf.if
    )";

    REQUIRE(Check(*ast, pattern));
}

TEST_CASE("Yield - single value", "[AST]") {
    const auto yieldExpr = ast::yield({ ast::constant(1.0f) });
    const auto ast = ast::module_({
        ast::function("fun",
                      {},
                      {},
                      {
                          ast::if_(ast::constant(true),
                                   { yieldExpr },
                                   { yieldExpr }),
                          ast::return_(),
                      }),
    });

    const auto pattern = R"(
        // CHECK: scf.yield %[[V1:.*]] : f32
    )";

    REQUIRE(Check(*ast, pattern));
}

TEST_CASE("Yield - expression unpacking", "[AST]") {
    const auto yieldExpr = ast::yield({
        ast::call("mrv", {}),
        ast::constant(int32_t(1)),
    });
    const auto ast = ast::module_({
        ast::function("mrv",
                      {},
                      { ast::ScalarType::FLOAT32, ast::ScalarType::FLOAT32 },
                      { ast::return_({ ast::constant(1.0f), ast::constant(1.0f) }) }),
        ast::function("fun",
                      {},
                      {},
                      { ast::if_(ast::constant(true),
                                 { yieldExpr },
                                 { yieldExpr }),
                        ast::return_() }),
    });

    const auto pattern = R"(
        // CHECK: scf.yield %[[F1:.*]], %[[F2:.*]], %[[I1:.*]] : f32, f32, i32
    )";

    REQUIRE(Check(*ast, pattern));
}