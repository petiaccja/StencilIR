#include <TestTools/FileCheck.hpp>

#include <IR/Ops.hpp>

#include <catch2/catch.hpp>

using namespace sir;


TEST_CASE("If", "[DAG]") {
    auto mod = ops::ModuleOp();
    auto func = mod.Create<ops::FuncOp>("fn", FunctionType::Get({ Bool, Float32, Float32 }, { Float32 }));

    auto ifop = func.Create<ops::IfOp>(func.GetRegionArg(0), 1);
    ifop.GetThenRegion().Create<ops::YieldOp>(std::vector<Value>{ func.GetRegionArg(1) });
    ifop.GetElseRegion().Create<ops::YieldOp>(std::vector<Value>{ func.GetRegionArg(2) });
    func.Create<ops::ReturnOp>(std::vector{ ifop.GetResults()[0] });

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
    auto mod = ops::ModuleOp();
    auto func = mod.Create<ops::FuncOp>("fn", FunctionType::Get({ IndexType::Get(), IndexType::Get(), IndexType::Get(), Float32 }, { Float32 }));

    auto forop = func.Create<ops::ForOp>(func.GetRegionArg(0), func.GetRegionArg(1), func.GetRegionArg(2),
                                         std::vector{ func.GetRegionArg(3) });
    forop.GetBody().Create<ops::YieldOp>(std::vector<Value>{ forop.GetRegionArg(1) });
    func.Create<ops::ReturnOp>(std::vector{ forop.GetResults()[0] });

    const auto pattern = R"(
        // CHECK: func @fn(%[[START:.*]]: index, %[[STOP:.*]]: index, %[[STEP:.*]]: index, %[[INIT:.*]]: f32) -> f32
        // CHECK-NEXT: %[[RES:.*]] = scf.for %[[IDX:.*]] = %[[START]] to %[[STOP]] step %[[STEP]] iter_args(%[[C:.*]] = %[[INIT]]) -> (f32) {
        // CHECK-NEXT:   scf.yield %[[C]] : f32
        // CHECK-NEXT: }
        // CHECK-NEXT: return %[[RES]]
    )";

    REQUIRE(CheckDAG(mod, pattern));
}
