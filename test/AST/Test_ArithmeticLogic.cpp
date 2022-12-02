#include <TestTools/ASTUtils.hpp>
#include <TestTools/FileCheck.hpp>

#include <AST/Building.hpp>

#include <catch2/catch.hpp>


TEST_CASE("Min", "[AST]") {
    const auto ast = EncloseInFunction(ast::min(ast::constant(1.0f), ast::constant(2.0f)));

    const auto pattern = R"(
        // CHECK: minf
    )";

    REQUIRE(CheckAST(*ast, pattern));
}

TEST_CASE("Max", "[AST]") {
    const auto ast = EncloseInFunction(ast::max(ast::constant(1), ast::constant(2)));

    const auto pattern = R"(
        // CHECK: maxsi
    )";

    REQUIRE(CheckAST(*ast, pattern));
}