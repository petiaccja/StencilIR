#include <TestTools/ASTUtils.hpp>
#include <TestTools/FileCheck.hpp>

#include <AST/Building.hpp>

#include <catch2/catch.hpp>


TEST_CASE("Block - without return value", "[AST]") {
    const auto ast = EncloseStatements(
        ast::block({
            ast::constant(1.0f),
            ast::yield(),
        }));

    const auto pattern = R"(
        // CHECK: scf.execute_region
    )";

    REQUIRE(CheckAST(*ast, pattern));
}

TEST_CASE("Block - with return value", "[AST]") {
    const auto ast = EncloseStatements(
        ast::block({
            ast::constant(1.0f),
            ast::yield({ ast::constant(2.0f) }),
        }));

    const auto pattern = R"(
        // CHECK: %[[RESULT:.*]] = scf.execute_region
    )";

    REQUIRE(CheckAST(*ast, pattern));
}