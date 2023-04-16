#include <TestTools/FileCheck.hpp>

#include <IR/Ops.hpp>

#include <catch2/catch.hpp>


TEST_CASE("If", "[DAG]") {
    auto mod = dag::ModuleOp();
    auto func = mod.Create<dag::FuncOp>("fn", ast::FunctionType::Get({ ast::Bool, ast::Float32, ast::Float32 }, { ast::Float32 }));

    auto ifop = func.Create<dag::IfOp>(func.GetRegionArg(0), 1);
    ifop.GetThenRegion().Create<dag::YieldOp>(std::vector<dag::Value>{ func.GetRegionArg(1) });
    ifop.GetElseRegion().Create<dag::YieldOp>(std::vector<dag::Value>{ func.GetRegionArg(2) });
    func.Create<dag::ReturnOp>(std::vector{ ifop.GetResults()[0] });

    const auto pattern = R"(
        // CHECK: func @fn(%[[COND:.*]]: i1, %[[TV:.*]]: f32, %[[FV:.*]]: f32) -> f32
        // CHECK-NEXT: %[[RES:.*]] = scf.if %[[COND]] -> (f32) {
        // CHECK-NEXT:   scf.yield %[[TV]] : f32
        // CHECK-NEXT: } else {
        // CHECK-NEXT:   scf.yield %[[FV]] : f32
        // CHECK-NEXT: }
        // CHECK-NEXT: return %[[RES]]
    )";

    REQUIRE(CheckDAG(mod, pattern));
}


TEST_CASE("For", "[DAG]") {
    auto mod = dag::ModuleOp();
    auto func = mod.Create<dag::FuncOp>("fn", ast::FunctionType::Get({ ast::IndexType::Get(), ast::IndexType::Get(), ast::IndexType::Get(), ast::Float32 }, { ast::Float32 }));

    auto forop = func.Create<dag::ForOp>(func.GetRegionArg(0), func.GetRegionArg(1), func.GetRegionArg(2),
                                         std::vector{ func.GetRegionArg(3) });
    forop.GetBody().Create<dag::YieldOp>(std::vector<dag::Value>{ forop.GetRegionArg(1) });
    func.Create<dag::ReturnOp>(std::vector{ forop.GetResults()[0] });

    const auto pattern = R"(
        // CHECK: func @fn(%[[START:.*]]: index, %[[STOP:.*]]: index, %[[STEP:.*]]: index, %[[INIT:.*]]: f32) -> f32
        // CHECK-NEXT: %[[RES:.*]] = scf.for %[[IDX:.*]] = %[[START]] to %[[STOP]] step %[[STEP]] iter_args(%[[C:.*]] = %[[INIT]]) -> (f32) {
        // CHECK-NEXT:   scf.yield %[[C]] : f32
        // CHECK-NEXT: }
        // CHECK-NEXT: return %[[RES]]
    )";

    REQUIRE(CheckDAG(mod, pattern));
}
