#include <TestTools/FileCheck.hpp>

#include <IR/Ops.hpp>

#include <catch2/catch.hpp>


using namespace sir;


TEST_CASE("Arithmetic cast", "[DAG]") {
    const auto funcType = FunctionType::Get({ FloatType::Get(32) }, { FloatType::Get(64) });

    auto mod = ops::ModuleOp();
    auto func = mod.Create<ops::FuncOp>("fn", funcType);

    SECTION("Cast") {
        auto cast = func.Create<ops::CastOp>(func.GetRegionArg(0), FloatType::Get(64));
        func.Create<ops::ReturnOp>(std::vector{ cast.GetResult() });

        const auto pattern = R"(
            // CHECK: func @fn(%[[ARG:.*]]: f32) -> f64
            // CHECK-NEXT: %[[OUT:.*]] = arith.extf %[[ARG]] : f32 to f64
        )";

        REQUIRE(CheckDAG(mod, pattern));
    }
}


TEST_CASE("Arithmetic constant", "[DAG]") {
    const auto funcType = FunctionType::Get({}, { FloatType::Get(32) });

    auto mod = ops::ModuleOp();
    auto func = mod.Create<ops::FuncOp>("fn", funcType);

    SECTION("Constant") {
        auto constant = func.Create<ops::ConstantOp>(1.0f, FloatType::Get(32));
        func.Create<ops::ReturnOp>(std::vector{ constant.GetResult() });

        const auto pattern = R"(
            // CHECK: func @fn() -> f32
            // CHECK-NEXT: %[[OUT:.*]] = arith.constant 1.000000e+00 : f32
            // CHECK-NEXT: return %[[OUT]]
        )";

        REQUIRE(CheckDAG(mod, pattern));
    }
}


TEST_CASE("Arithmetic binary", "[DAG]") {
    const auto funcType = FunctionType::Get({ FloatType::Get(32), FloatType::Get(32) }, { FloatType::Get(32) });

    auto mod = ops::ModuleOp();
    auto func = mod.Create<ops::FuncOp>("fn", funcType);

    SECTION("Add") {
        auto add = func.Create<ops::ArithmeticOp>(func.GetRegionArg(0), func.GetRegionArg(1), ops::eArithmeticFunction::ADD);
        func.Create<ops::ReturnOp>(std::vector{ add.GetResult() });

        const auto pattern = R"(
            // CHECK: func @fn(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32) -> f32
            // CHECK-NEXT: %[[OUT:.*]] = arith.addf %[[LHS]], %[[RHS]] : f32
            // CHECK-NEXT: return %[[OUT]]
        )";

        REQUIRE(CheckDAG(mod, pattern));
    }
    SECTION("Min") {
        auto min = func.Create<ops::MinOp>(func.GetRegionArg(0), func.GetRegionArg(1));
        func.Create<ops::ReturnOp>(std::vector{ min.GetResult() });

        const auto pattern = R"(
            // CHECK: func @fn(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32) -> f32
            // CHECK-NEXT: %[[OUT:.*]] = arith.minf %[[LHS]], %[[RHS]] : f32
            // CHECK-NEXT: return %[[OUT]]
        )";

        REQUIRE(CheckDAG(mod, pattern));
    }
    SECTION("Max") {
        auto max = func.Create<ops::MaxOp>(func.GetRegionArg(0), func.GetRegionArg(1));
        func.Create<ops::ReturnOp>(std::vector{ max.GetResult() });

        const auto pattern = R"(
            // CHECK: func @fn(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32) -> f32
            // CHECK-NEXT: %[[OUT:.*]] = arith.maxf %[[LHS]], %[[RHS]] : f32
            // CHECK-NEXT: return %[[OUT]]
        )";

        REQUIRE(CheckDAG(mod, pattern));
    }
}


TEST_CASE("Logic binary", "[DAG]") {
    const auto funcType = FunctionType::Get({ FloatType::Get(32), FloatType::Get(32) }, { IntegerType::Get(1, true) });

    auto mod = ops::ModuleOp();
    auto func = mod.Create<ops::FuncOp>("fn", funcType);

    SECTION("Less") {
        auto lt = func.Create<ops::ComparisonOp>(func.GetRegionArg(0), func.GetRegionArg(1), ops::eComparisonFunction::LT);
        func.Create<ops::ReturnOp>(std::vector{ lt.GetResult() });

        const auto pattern = R"(
            // CHECK: func @fn(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32) -> i1
            // CHECK-NEXT: %[[OUT:.*]] = arith.cmpf olt, %[[LHS]], %[[RHS]] : f32
            // CHECK-NEXT: return %[[OUT]]
        )";

        REQUIRE(CheckDAG(mod, pattern));
    }
}