#include <TestTools/ASTUtils.hpp>
#include <TestTools/FileCheck.hpp>

#include <AST/Building.hpp>

#include <catch2/catch.hpp>



TEST_CASE("Project", "[AST]") {
    const auto ast = EncloseInStencil<3>(ast::project(ast::index(), { 0, 0, 2, 1 }));

    const auto pattern = R"(
        // CHECK: %[[R:.*]] = project %[[IDX:.*]][0, 0, 2, 1]
    )";

    REQUIRE(CheckAST(*ast, pattern));
}


TEST_CASE("Extend", "[AST]") {
    const auto ast = EncloseInStencil<3>(ast::extend(ast::index(), 1, ast::constant(0, ast::IndexType::Get())));

    const auto pattern = R"(
        // CHECK: %[[R:.*]] = extend %[[IDX:.*]][1], %[[V:.*]]
    )";

    REQUIRE(CheckAST(*ast, pattern));
}


TEST_CASE("Exchange", "[AST]") {
    const auto ast = EncloseInStencil<3>(ast::exchange(ast::index(), 1, ast::constant(0, ast::IndexType::Get())));

    const auto pattern = R"(
        // CHECK: %[[R:.*]] = exchange %[[IDX:.*]][1], %[[V:.*]]
    )";

    REQUIRE(CheckAST(*ast, pattern));
}


TEST_CASE("Extract", "[AST]") {
    const auto ast = EncloseInStencil<3>(ast::extract(ast::index(), 1));

    const auto pattern = R"(
        // CHECK: %[[R:.*]] = extract %[[IDX:.*]][1]
    )";

    REQUIRE(CheckAST(*ast, pattern));
}