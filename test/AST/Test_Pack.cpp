#include <TestTools/FileCheck.hpp>

#include <AST/Building.hpp>

#include <catch2/catch.hpp>


TEST_CASE("Pack - expression packing", "[AST]") {
    const auto ast = ast::module_({
        ast::function("mrv",
                      {},
                      { ast::FloatType::Get(32), ast::FloatType::Get(32) },
                      { ast::return_({ ast::constant(1.0f), ast::constant(1.0f) }) }),
        ast::function("fun",
                      {},
                      { ast::FloatType::Get(32), ast::FloatType::Get(32), ast::FloatType::Get(32) },
                      {
                          ast::return_({ ast::pack({ ast::call("mrv", {}), ast::constant(1.0f) }) }),
                      }),
    });

    const auto pattern = R"(
        // CHECK: return %[[R1:.*]], %[[R2:.*]], %[[R3:.*]]
    )";

    REQUIRE(CheckAST(*ast, pattern));
}