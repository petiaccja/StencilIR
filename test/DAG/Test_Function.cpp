#include <DAG/Ops.hpp>
#include <TestTools/FileCheck.hpp>

#include <catch2/catch.hpp>


TEST_CASE("Function: create", "[AST]") {
    auto mod = dag::ModuleOp();
    mod.Create<dag::FuncOp>("funcname", ast::FunctionType::Get({ ast::FloatType::Get(32) }, { ast::FloatType::Get(32) }));

    const auto pattern = R"(
        // CHECK: func @funcname(%[[ARG:.*]]: f32) -> f32
        // CHECK-NEXT: return %[[ARG]]
    )";

    REQUIRE(CheckDAG(mod, pattern));
}