#include "Utility/RunModule.hpp"

#include <Compiler/Pipelines.hpp>
#include <Execution/Execution.hpp>
#include <IR/Ops.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string_view>

#include <catch2/catch.hpp>

using namespace sir;


static ops::ModuleOp CreateAST() {
    auto moduleOp = ops::ModuleOp{};

    auto snSubstract = moduleOp.Create<ops::StencilOp>(
        "subtract",
        ast::FunctionType::Get({ ast::FieldType::Get(ast::Float32, 1),
                                 ast::FieldType::Get(ast::Float32, 1) },
                               { ast::Float32 }),
        1);
    auto idx = snSubstract.Create<ops::IndexOp>().GetResult();
    auto lSample = snSubstract.Create<ops::SampleOp>(snSubstract.GetRegionArg(0), idx).GetResult();
    auto rSample = snSubstract.Create<ops::SampleOp>(snSubstract.GetRegionArg(1), idx).GetResult();
    auto sum = snSubstract.Create<ops::ArithmeticOp>(lSample, rSample, ops::eArithmeticFunction::SUB)
                   .GetResult();
    snSubstract.Create<ops::ReturnOp>(std::vector{ sum });


    auto fnMain = moduleOp.Create<ops::FuncOp>(
        "main",
        ast::FunctionType::Get({ ast::FieldType::Get(ast::Float32, 1), ast::FieldType::Get(ast::Float32, 1) }, {}));


    auto input = fnMain.GetRegionArg(0);
    auto output = fnMain.GetRegionArg(1);
    auto czero = fnMain.Create<ops::ConstantOp>(0, ast::IndexType::Get()).GetResult();
    auto cone = fnMain.Create<ops::ConstantOp>(1, ast::IndexType::Get()).GetResult();
    auto size = fnMain.Create<ops::DimOp>(input, czero).GetResult();
    auto dsize = fnMain.Create<ops::ArithmeticOp>(size, cone, ops::eArithmeticFunction::SUB).GetResult();
    auto ddsize = fnMain.Create<ops::ArithmeticOp>(dsize, cone, ops::eArithmeticFunction::SUB).GetResult();


    auto left = fnMain.Create<ops::ExtractSliceOp>(input, std::vector{ czero }, std::vector{ dsize }, std::vector{ cone }).GetResult();
    auto right = fnMain.Create<ops::ExtractSliceOp>(input, std::vector{ cone }, std::vector{ dsize }, std::vector{ cone }).GetResult();
    auto tmp1 = fnMain.Create<ops::AllocTensorOp>(ast::Float32, std::vector{ dsize }).GetResult();

    auto d = fnMain.Create<ops::ApplyOp>("subtract",
                                         std::vector{ left, right },
                                         std::vector{ tmp1 },
                                         std::vector<Value>{},
                                         std::vector<int64_t>{ 0, 0 })
                 .GetResults()[0];

    auto dleft = fnMain.Create<ops::ExtractSliceOp>(d, std::vector{ czero }, std::vector{ ddsize }, std::vector{ cone }).GetResult();
    auto dright = fnMain.Create<ops::ExtractSliceOp>(d, std::vector{ cone }, std::vector{ ddsize }, std::vector{ cone }).GetResult();

    auto dd = fnMain.Create<ops::ApplyOp>("subtract",
                                          std::vector{ dleft, dright },
                                          std::vector{ output },
                                          std::vector<Value>{},
                                          std::vector<int64_t>{ 0, 0 })
                  .GetResults()[0];

    fnMain.Create<ops::ReturnOp>(std::vector<Value>{});

    return moduleOp;
}

TEST_CASE("Optimization", "[Program]") {
    constexpr ptrdiff_t inputSize = 9;
    constexpr ptrdiff_t outputSize = inputSize - 2;
    std::array<float, inputSize> inputBuffer;
    std::array<float, outputSize> outputBuffer;
    StridedMemRefType<float, 1> input{ inputBuffer.data(), inputBuffer.data(), 0, { inputSize }, { 1 } };
    StridedMemRefType<float, 1> output{ outputBuffer.data(), outputBuffer.data(), 0, { outputSize }, { 1 } };
    std::ranges::fill(outputBuffer, 0);

    for (size_t x = 0; x < inputSize; ++x) {
        inputBuffer[x] = float(x * x * x) * 0.1f;
    }

    const auto program = CreateAST();
    const auto stages = RunModule(program, "main", true, input, output);

    const std::array<float, outputSize> expectedBuffer = {
        0.6f, 1.2f, 1.8f, 2.4f, 3.0f, 3.6f, 4.2f
    };

    const auto maxDifference = std::inner_product(
        outputBuffer.begin(),
        outputBuffer.end(),
        expectedBuffer.begin(),
        0.0f,
        [](float acc, float v) { return std::max(acc, v); },
        [](float u, float v) { return std::abs(u - v); });
    std::stringstream ss;
    for (auto& stage : stages) {
        ss << "// " << stage.name << std::endl;
        ss << stage.ir << "\n"
           << std::endl;
    }
    INFO(ss.str());
    REQUIRE(maxDifference < 0.001f);
}