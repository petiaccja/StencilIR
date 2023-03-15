#include <TestTools/FileCheck.hpp>

#include <DAG/Ops.hpp>

#include <catch2/catch.hpp>


TEST_CASE("Arithmetic cast", "[DAG]") {
    const auto funcType = ast::FunctionType::Get({ ast::FloatType::Get(32) }, { ast::FloatType::Get(64) });

    auto mod = dag::ModuleOp();
    auto func = mod.Create<dag::FuncOp>("fn", funcType);

    SECTION("Cast") {
        auto cast = func.Create<dag::CastOp>(func.GetRegionArg(0), ast::FloatType::Get(64));
        func.Create<dag::ReturnOp>(std::vector{ cast.GetResult() });

        const auto pattern = R"(
            // CHECK: func @fn(%[[ARG:.*]]: f32) -> f64
            // CHECK-NEXT: %[[OUT:.*]] = arith.extf %[[ARG]] : f32 to f64
            // CHECK-NEXT: return %[[OUT]]
        )";

        REQUIRE(CheckDAG(mod, pattern));
    }
}


TEST_CASE("Arithmetic constant", "[DAG]") {
    const auto funcType = ast::FunctionType::Get({}, { ast::FloatType::Get(32) });

    auto mod = dag::ModuleOp();
    auto func = mod.Create<dag::FuncOp>("fn", funcType);

    SECTION("Constant") {
        auto constant = func.Create<dag::ConstantOp>(1.0f, ast::FloatType::Get(32));
        func.Create<dag::ReturnOp>(std::vector{ constant.GetResult() });

        const auto pattern = R"(
            // CHECK: func @fn() -> f32
            // CHECK-NEXT: %[[OUT:.*]] = arith.constant 1.000000e+00 : f32
            // CHECK-NEXT: return %[[OUT]]
        )";

        REQUIRE(CheckDAG(mod, pattern));
    }
}


TEST_CASE("Arithmetic binary", "[DAG]") {
    const auto funcType = ast::FunctionType::Get({ ast::FloatType::Get(32), ast::FloatType::Get(32) }, { ast::FloatType::Get(32) });

    auto mod = dag::ModuleOp();
    auto func = mod.Create<dag::FuncOp>("fn", funcType);

    SECTION("Add") {
        auto add = func.Create<dag::ArithmeticOp>(func.GetRegionArg(0), func.GetRegionArg(1), dag::eArithmeticFunction::ADD);
        func.Create<dag::ReturnOp>(std::vector{ add.GetResult() });

        const auto pattern = R"(
            // CHECK: func @fn(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32) -> f32
            // CHECK-NEXT: %[[OUT:.*]] = arith.addf %[[LHS]], %[[RHS]] : f32
            // CHECK-NEXT: return %[[OUT]]
        )";

        REQUIRE(CheckDAG(mod, pattern));
    }
    SECTION("Min") {
        auto min = func.Create<dag::MinOp>(func.GetRegionArg(0), func.GetRegionArg(1));
        func.Create<dag::ReturnOp>(std::vector{ min.GetResult() });

        const auto pattern = R"(
            // CHECK: func @fn(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32) -> f32
            // CHECK-NEXT: %[[OUT:.*]] = arith.minf %[[LHS]], %[[RHS]] : f32
            // CHECK-NEXT: return %[[OUT]]
        )";

        REQUIRE(CheckDAG(mod, pattern));
    }
    SECTION("Max") {
        auto max = func.Create<dag::MaxOp>(func.GetRegionArg(0), func.GetRegionArg(1));
        func.Create<dag::ReturnOp>(std::vector{ max.GetResult() });

        const auto pattern = R"(
            // CHECK: func @fn(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32) -> f32
            // CHECK-NEXT: %[[OUT:.*]] = arith.maxf %[[LHS]], %[[RHS]] : f32
            // CHECK-NEXT: return %[[OUT]]
        )";

        REQUIRE(CheckDAG(mod, pattern));
    }
}


TEST_CASE("Logic binary", "[DAG]") {
    const auto funcType = ast::FunctionType::Get({ ast::FloatType::Get(32), ast::FloatType::Get(32) }, { ast::IntegerType::Get(1, true) });

    auto mod = dag::ModuleOp();
    auto func = mod.Create<dag::FuncOp>("fn", funcType);

    SECTION("Less") {
        auto lt = func.Create<dag::ComparisonOp>(func.GetRegionArg(0), func.GetRegionArg(1), dag::eComparisonFunction::LT);
        func.Create<dag::ReturnOp>(std::vector{ lt.GetResult() });

        const auto pattern = R"(
            // CHECK: func @fn(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32) -> i1
            // CHECK-NEXT: %[[OUT:.*]] = arith.cmpf olt, %[[LHS]], %[[RHS]] : f32
            // CHECK-NEXT: return %[[OUT]]
        )";

        REQUIRE(CheckDAG(mod, pattern));
    }
}