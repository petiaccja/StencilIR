#include <DAG/Ops.hpp>
#include <TestTools/FileCheck.hpp>

#include <catch2/catch.hpp>


TEST_CASE("Index", "[DAG]") {
    auto mod = dag::ModuleOp();
    auto stencil = mod.Create<dag::StencilOp>("sn", ast::FunctionType::Get({}, {}), 2);
    stencil.Create<dag::IndexOp>();
    stencil.Create<dag::ReturnOp>(std::vector<dag::Value>{});

    const auto pattern = R"(
        // CHECK: stencil.stencil @sn()
        // CHECK-NEXT: %[[IDX:.*]] = index : vector<2xindex>
        // CHECK-NEXT: return
    )";

    REQUIRE(CheckDAG(mod, pattern));
}


TEST_CASE("Jump", "[DAG]") {
    auto mod = dag::ModuleOp();
    auto stencil = mod.Create<dag::StencilOp>("sn", ast::FunctionType::Get({}, {}), 2);
    auto idx = stencil.Create<dag::IndexOp>();
    stencil.Create<dag::JumpOp>(idx.GetResult(), std::vector<int64_t>{ 2, 3 });
    stencil.Create<dag::ReturnOp>(std::vector<dag::Value>{});

    const auto pattern = R"(
        // CHECK: stencil.stencil @sn()
        // CHECK-NEXT: %[[IDX:.*]] = index : vector<2xindex>
        // CHECK-NEXT: %[[MOD:.*]] = jump %[[IDX]], [2, 3] : (vector<2xindex>) -> vector<2xindex>
        // CHECK-NEXT: return
    )";

    REQUIRE(CheckDAG(mod, pattern));
}


TEST_CASE("Project", "[DAG]") {
    auto mod = dag::ModuleOp();
    auto stencil = mod.Create<dag::StencilOp>("sn", ast::FunctionType::Get({}, {}), 2);
    auto idx = stencil.Create<dag::IndexOp>();
    stencil.Create<dag::ProjectOp>(idx.GetResult(), std::vector<int64_t>{ 2, 3 });
    stencil.Create<dag::ReturnOp>(std::vector<dag::Value>{});

    const auto pattern = R"(
        // CHECK: stencil.stencil @sn()
        // CHECK-NEXT: %[[IDX:.*]] = index : vector<2xindex>
        // CHECK-NEXT: %[[MOD:.*]] = project %[[IDX]][2, 3] : (vector<2xindex>) -> vector<2xindex>
        // CHECK-NEXT: return
    )";

    REQUIRE(CheckDAG(mod, pattern));
}


TEST_CASE("Extract", "[DAG]") {
    auto mod = dag::ModuleOp();
    auto stencil = mod.Create<dag::StencilOp>("sn", ast::FunctionType::Get({}, {}), 2);
    auto idx = stencil.Create<dag::IndexOp>();
    stencil.Create<dag::ExtractOp>(idx.GetResult(), 1);
    stencil.Create<dag::ReturnOp>(std::vector<dag::Value>{});

    const auto pattern = R"(
        // CHECK: stencil.stencil @sn()
        // CHECK-NEXT: %[[IDX:.*]] = index : vector<2xindex>
        // CHECK-NEXT: %[[MOD:.*]] = extract %[[IDX]][1] : vector<2xindex>
        // CHECK-NEXT: return
    )";

    REQUIRE(CheckDAG(mod, pattern));
}


TEST_CASE("Extend", "[DAG]") {
    auto mod = dag::ModuleOp();
    auto stencil = mod.Create<dag::StencilOp>("sn", ast::FunctionType::Get({ ast::IndexType::Get() }, {}), 2);
    auto idx = stencil.Create<dag::IndexOp>();
    stencil.Create<dag::ExtendOp>(idx.GetResult(), 1, stencil.GetRegionArg(0));
    stencil.Create<dag::ReturnOp>(std::vector<dag::Value>{});

    const auto pattern = R"(
        // CHECK: stencil.stencil @sn(%[[VALUE:.*]]: index)
        // CHECK-NEXT: %[[IDX:.*]] = index : vector<2xindex>
        // CHECK-NEXT: %[[MOD:.*]] = extend %[[IDX]][1], %[[VALUE]] : (vector<2xindex>) -> vector<3xindex>
        // CHECK-NEXT: return
    )";

    REQUIRE(CheckDAG(mod, pattern));
}


TEST_CASE("Exchange", "[DAG]") {
    auto mod = dag::ModuleOp();
    auto stencil = mod.Create<dag::StencilOp>("sn", ast::FunctionType::Get({ ast::IndexType::Get() }, {}), 2);
    auto idx = stencil.Create<dag::IndexOp>();
    stencil.Create<dag::ExchangeOp>(idx.GetResult(), 1, stencil.GetRegionArg(0));
    stencil.Create<dag::ReturnOp>(std::vector<dag::Value>{});

    const auto pattern = R"(
        // CHECK: stencil.stencil @sn(%[[VALUE:.*]]: index)
        // CHECK-NEXT: %[[IDX:.*]] = index : vector<2xindex>
        // CHECK-NEXT: %[[MOD:.*]] = exchange %[[IDX]][1], %[[VALUE]] : (vector<2xindex>) -> vector<2xindex>
        // CHECK-NEXT: return
    )";

    REQUIRE(CheckDAG(mod, pattern));
}


TEST_CASE("Sample", "[DAG]") {
    auto mod = dag::ModuleOp();
    auto stencil = mod.Create<dag::StencilOp>("sn", ast::FunctionType::Get({ ast::FieldType::Get(ast::Float32, 2) }, {}), 2);
    auto idx = stencil.Create<dag::IndexOp>();
    stencil.Create<dag::SampleOp>(stencil.GetRegionArg(0), idx.GetResult());
    stencil.Create<dag::ReturnOp>(std::vector<dag::Value>{});

    const auto pattern = R"(
        // CHECK: stencil.stencil @sn(%[[FIELD:.*]]: tensor<?x?xf32>)
        // CHECK-NEXT: %[[IDX:.*]] = index : vector<2xindex>
        // CHECK-NEXT: %[[MOD:.*]] = sample %[[FIELD]][%[[IDX]]] : (tensor<?x?xf32>, vector<2xindex>) -> f32
        // CHECK-NEXT: return
    )";

    REQUIRE(CheckDAG(mod, pattern));
}