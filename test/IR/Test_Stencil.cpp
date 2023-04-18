#include <TestTools/FileCheck.hpp>

#include <IR/Ops.hpp>

#include <catch2/catch.hpp>

using namespace sir;


TEST_CASE("Index", "[DAG]") {
    auto mod = ops::ModuleOp();
    auto stencil = mod.Create<ops::StencilOp>("sn", FunctionType::Get({}, {}), 2);
    stencil.Create<ops::IndexOp>();
    stencil.Create<ops::ReturnOp>(std::vector<Value>{});

    const auto pattern = R"(
        // CHECK: stencil.stencil @sn()
        // CHECK-NEXT: %[[IDX:.*]] = index : vector<2xindex>
        // CHECK-NEXT: return
    )";

    REQUIRE(CheckDAG(mod, pattern));
}


TEST_CASE("Jump", "[DAG]") {
    auto mod = ops::ModuleOp();
    auto stencil = mod.Create<ops::StencilOp>("sn", FunctionType::Get({}, {}), 2);
    auto idx = stencil.Create<ops::IndexOp>();
    stencil.Create<ops::JumpOp>(idx.GetResult(), std::vector<int64_t>{ 2, 3 });
    stencil.Create<ops::ReturnOp>(std::vector<Value>{});

    const auto pattern = R"(
        // CHECK: stencil.stencil @sn()
        // CHECK-NEXT: %[[IDX:.*]] = index : vector<2xindex>
        // CHECK-NEXT: %[[MOD:.*]] = jump %[[IDX]], [2, 3] : (vector<2xindex>) -> vector<2xindex>
        // CHECK-NEXT: return
    )";

    REQUIRE(CheckDAG(mod, pattern));
}


TEST_CASE("Project", "[DAG]") {
    auto mod = ops::ModuleOp();
    auto stencil = mod.Create<ops::StencilOp>("sn", FunctionType::Get({}, {}), 2);
    auto idx = stencil.Create<ops::IndexOp>();
    stencil.Create<ops::ProjectOp>(idx.GetResult(), std::vector<int64_t>{ 2, 3 });
    stencil.Create<ops::ReturnOp>(std::vector<Value>{});

    const auto pattern = R"(
        // CHECK: stencil.stencil @sn()
        // CHECK-NEXT: %[[IDX:.*]] = index : vector<2xindex>
        // CHECK-NEXT: %[[MOD:.*]] = project %[[IDX]][2, 3] : (vector<2xindex>) -> vector<2xindex>
        // CHECK-NEXT: return
    )";

    REQUIRE(CheckDAG(mod, pattern));
}


TEST_CASE("Extract", "[DAG]") {
    auto mod = ops::ModuleOp();
    auto stencil = mod.Create<ops::StencilOp>("sn", FunctionType::Get({}, {}), 2);
    auto idx = stencil.Create<ops::IndexOp>();
    stencil.Create<ops::ExtractOp>(idx.GetResult(), 1);
    stencil.Create<ops::ReturnOp>(std::vector<Value>{});

    const auto pattern = R"(
        // CHECK: stencil.stencil @sn()
        // CHECK-NEXT: %[[IDX:.*]] = index : vector<2xindex>
        // CHECK-NEXT: %[[MOD:.*]] = extract %[[IDX]][1] : vector<2xindex>
        // CHECK-NEXT: return
    )";

    REQUIRE(CheckDAG(mod, pattern));
}


TEST_CASE("Extend", "[DAG]") {
    auto mod = ops::ModuleOp();
    auto stencil = mod.Create<ops::StencilOp>("sn", FunctionType::Get({ IndexType::Get() }, {}), 2);
    auto idx = stencil.Create<ops::IndexOp>();
    stencil.Create<ops::ExtendOp>(idx.GetResult(), 1, stencil.GetRegionArg(0));
    stencil.Create<ops::ReturnOp>(std::vector<Value>{});

    const auto pattern = R"(
        // CHECK: stencil.stencil @sn(%[[VALUE:.*]]: index)
        // CHECK-NEXT: %[[IDX:.*]] = index : vector<2xindex>
        // CHECK-NEXT: %[[MOD:.*]] = extend %[[IDX]][1], %[[VALUE]] : (vector<2xindex>) -> vector<3xindex>
        // CHECK-NEXT: return
    )";

    REQUIRE(CheckDAG(mod, pattern));
}


TEST_CASE("Exchange", "[DAG]") {
    auto mod = ops::ModuleOp();
    auto stencil = mod.Create<ops::StencilOp>("sn", FunctionType::Get({ IndexType::Get() }, {}), 2);
    auto idx = stencil.Create<ops::IndexOp>();
    stencil.Create<ops::ExchangeOp>(idx.GetResult(), 1, stencil.GetRegionArg(0));
    stencil.Create<ops::ReturnOp>(std::vector<Value>{});

    const auto pattern = R"(
        // CHECK: stencil.stencil @sn(%[[VALUE:.*]]: index)
        // CHECK-NEXT: %[[IDX:.*]] = index : vector<2xindex>
        // CHECK-NEXT: %[[MOD:.*]] = exchange %[[IDX]][1], %[[VALUE]] : (vector<2xindex>) -> vector<2xindex>
        // CHECK-NEXT: return
    )";

    REQUIRE(CheckDAG(mod, pattern));
}


TEST_CASE("Sample", "[DAG]") {
    auto mod = ops::ModuleOp();
    auto stencil = mod.Create<ops::StencilOp>("sn", FunctionType::Get({ FieldType::Get(Float32, 2) }, {}), 2);
    auto idx = stencil.Create<ops::IndexOp>();
    stencil.Create<ops::SampleOp>(stencil.GetRegionArg(0), idx.GetResult());
    stencil.Create<ops::ReturnOp>(std::vector<Value>{});

    const auto pattern = R"(
        // CHECK: stencil.stencil @sn(%[[FIELD:.*]]: tensor<?x?xf32>)
        // CHECK-NEXT: %[[IDX:.*]] = index : vector<2xindex>
        // CHECK-NEXT: %[[MOD:.*]] = sample %[[FIELD]][%[[IDX]]] : (tensor<?x?xf32>, vector<2xindex>) -> f32
        // CHECK-NEXT: return
    )";

    REQUIRE(CheckDAG(mod, pattern));
}