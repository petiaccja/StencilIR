#include <TestTools/ASTUtils.hpp>
#include <TestTools/FileCheck.hpp>

#include <AST/Building.hpp>

#include <catch2/catch.hpp>


TEST_CASE("If - then block", "[AST]") {
    const auto ast = EncloseStatements(
        ast::if_(
            ast::constant(true),
            { ast::constant(int32_t(1)), ast::yield() },
            {}));

    const auto pattern = R"(
        // CHECK: %[[COND:.*]] = arith.constant true
        // CHECK-NEXT: scf.if %[[COND]] {
        // CHECK-NEXT: %[[V1:.*]] = arith.constant 1
        // CHECK-NEXT: }
        // CHECK-NEXT: return
    )";

    REQUIRE(CheckAST(*ast, pattern));
}

TEST_CASE("If - then + else block", "[AST]") {
    const auto ast = EncloseStatements(
        ast::if_(
            ast::constant(true),
            { ast::constant(int32_t(1)), ast::yield() },
            { ast::constant(int32_t(2)), ast::yield() }));

    const auto pattern = R"(
        // CHECK: %[[COND:.*]] = arith.constant true
        // CHECK-NEXT: scf.if %[[COND]] {
        // CHECK-NEXT:   %[[V1:.*]] = arith.constant 1
        // CHECK-NEXT: } else {
        // CHECK-NEXT:   %[[V1:.*]] = arith.constant 2
        // CHECK-NEXT: }
        // CHECK-NEXT: return
    )";

    REQUIRE(CheckAST(*ast, pattern));
}

TEST_CASE("If - yield", "[AST]") {
    const auto ast = EncloseStatements(
        ast::if_(
            ast::constant(true),
            { ast::yield({ ast::constant(int32_t(1)), ast::constant(int32_t(2)) }) },
            { ast::yield({ ast::constant(int32_t(3)), ast::constant(int32_t(4)) }) }));

    const auto pattern = R"(
        // CHECK: %[[COND:.*]] = arith.constant true
        // CHECK-NEXT: %[[RS:.*]] scf.if %[[COND]] -> (i32, i32) {
        // CHECK-NEXT:   %[[V1:.*]] = arith.constant 1
        // CHECK-NEXT:   %[[V2:.*]] = arith.constant 2
        // CHECK-NEXT:   yield %[[V1]], %[[V2]]
        // CHECK-NEXT: } else {
        // CHECK-NEXT:   %[[V3:.*]] = arith.constant 3
        // CHECK-NEXT:   %[[V4:.*]] = arith.constant 4
        // CHECK-NEXT:   yield %[[V3]], %[[V4]]
        // CHECK-NEXT: }
        // CHECK-NEXT: return
    )";

    REQUIRE(CheckAST(*ast, pattern));
}