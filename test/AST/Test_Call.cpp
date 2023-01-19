#include <TestTools/ASTUtils.hpp>
#include <TestTools/FileCheck.hpp>

#include <AST/Building.hpp>

#include <catch2/catch.hpp>


TEST_CASE("Call - regular", "[AST]") {
    const auto ast = ast::module_({
        ast::function("callee",
                      { ast::Parameter{ "a", ast::FloatType::Get(32) } },
                      { ast::FloatType::Get(32) },
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

    REQUIRE(CheckAST(*ast, pattern));
}

TEST_CASE("Call - expression unpacking", "[AST]") {
    const auto ast = ast::module_({
        ast::function("mrv",
                      {},
                      { ast::FloatType::Get(32), ast::FloatType::Get(32) },
                      { ast::return_({ ast::constant(1.0f), ast::constant(1.0f) }) }),
        ast::function("callee",
                      {
                          ast::Parameter{ "a", ast::FloatType::Get(32) },
                          ast::Parameter{ "b", ast::FloatType::Get(32) },
                          ast::Parameter{ "c", ast::FloatType::Get(32) },
                      },
                      {},
                      { ast::return_() }),
        ast::function("caller",
                      {},
                      {},
                      {
                          ast::call("callee", { ast::call("mrv", {}), ast::constant(1.0f) }),
                          ast::return_(),
                      }),
    });

    const auto pattern = R"(
        // CHECK: call @callee(%[[ARG1:.*]], %[[ARG2:.*]], %[[ARG3:.*]])
    )";

    REQUIRE(CheckAST(*ast, pattern));
}