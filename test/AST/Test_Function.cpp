#include <TestTools/FileCheck.hpp>

#include <AST/Building.hpp>

#include <catch2/catch.hpp>


TEST_CASE("Function: create", "[AST]") {
    const auto ast = ast::module_({
        ast::function("funcname",
                      { ast::Parameter{ "a", ast::FloatType::Get(32) } },
                      { ast::FloatType::Get(32) },
                      { ast::return_({ ast::symref("a") }) }),
    });

    const auto pattern = R"(
        // CHECK: func @funcname(%[[ARG:.*]]: f32) -> f32
        // CHECK-NEXT: return %[[ARG]]
    )";

    REQUIRE(CheckAST(*ast, pattern));
}