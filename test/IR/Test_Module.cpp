#include <TestTools/FileCheck.hpp>

#include <IR/Ops.hpp>

#include <catch2/catch.hpp>

using namespace sir;


TEST_CASE("Function", "[DAG]") {
    auto mod = ops::ModuleOp();
    auto func = mod.Create<ops::FuncOp>("fn", ast::FunctionType::Get({ ast::Float32 }, { ast::Float32 }));
    func.Create<ops::ReturnOp>(std::vector{ func.GetRegionArg(0) });

    const auto pattern = R"(
        // CHECK: func @fn(%[[ARG:.*]]: f32) -> f32
        // CHECK-NEXT: return %[[ARG]]
    )";

    REQUIRE(CheckDAG(mod, pattern));
}


TEST_CASE("Stencil", "[DAG]") {
    auto mod = ops::ModuleOp();
    auto stencil = mod.Create<ops::StencilOp>("sn", ast::FunctionType::Get({ ast::Float32 }, { ast::Float32 }), 2);
    stencil.Create<ops::ReturnOp>(std::vector{ stencil.GetRegionArg(0) });

    const auto pattern = R"(
        // CHECK: stencil.stencil @sn(%[[ARG:.*]]: f32) -> f32
        // CHECK-NEXT: return %[[ARG]]
    )";

    REQUIRE(CheckDAG(mod, pattern));
}


TEST_CASE("Call", "[DAG]") {
    auto mod = ops::ModuleOp();

    auto callee = mod.Create<ops::FuncOp>("callee", ast::FunctionType::Get({ ast::Float32 }, { ast::Float32 }));
    callee.Create<ops::ReturnOp>(std::vector{ callee.GetRegionArg(0) });

    auto caller = mod.Create<ops::FuncOp>("caller", ast::FunctionType::Get({ ast::Float32 }, { ast::Float32 }));
    auto call = caller.Create<ops::CallOp>(callee, std::vector{ caller.GetRegionArg(0) });
    caller.Create<ops::ReturnOp>(std::vector{ call.GetResults()[0] });

    const auto pattern = R"(
        // CHECK: func @caller(%[[ARG:.*]]: f32) -> f32
        // CHECK-NEXT: %[[OUT:.*]] = call @callee(%[[ARG]])
        // CHECK-NEXT: return %[[OUT]]
    )";

    REQUIRE(CheckDAG(mod, pattern));
}


TEST_CASE("Apply", "[DAG]") {
    auto mod = ops::ModuleOp();
    auto stencil = mod.Create<ops::StencilOp>("sn",
                                              ast::FunctionType::Get(
                                                  { ast::Float32 },
                                                  { ast::Float32 }),
                                              2);
    stencil.Create<ops::ReturnOp>(std::vector{ stencil.GetRegionArg(0) });


    auto func = mod.Create<ops::FuncOp>("fn",
                                        ast::FunctionType::Get(
                                            { ast::Float32, ast::FieldType::Get(ast::Float32, 2) },
                                            { ast::FieldType::Get(ast::Float32, 2) }));
    auto apply = func.Create<ops::ApplyOp>(stencil,
                                           std::vector{ func.GetRegionArg(0) },
                                           std::vector{ func.GetRegionArg(1) },
                                           std::vector<Value>{},
                                           std::vector<int64_t>{ 0, 0 });
    func.Create<ops::ReturnOp>(std::vector{ apply.GetResults()[0] });


    const auto pattern = R"(
        // CHECK: func @fn(%[[ARG:.*]]: f32, %[[OUT:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32>
        // CHECK-NEXT: %[[RES:.*]] = stencil.apply @sn(%[[ARG]]) outs(%[[OUT]])
        // CHECK-NEXT: return %[[RES]]
    )";

    REQUIRE(CheckDAG(mod, pattern));
}