#include "Utility/RunModule.hpp"

#include <Compiler/Pipelines.hpp>
#include <Diagnostics/Exception.hpp>
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



static ops::ModuleOp CreateModule() {
    const auto indexT = IndexType::Get();
    const auto scalarT = FloatType::Get(32);
    const auto fieldT = FieldType::Get(scalarT, 2);

    auto mod = ops::ModuleOp{};

    auto weigh = mod.Create<ops::StencilOp>("weigh", FunctionType::Get({ scalarT, fieldT }, { scalarT }), 2, false);
    {
        const auto weight = weigh.GetRegionArg(0);
        const auto data = weigh.GetRegionArg(1);
        const auto idx = weigh.Create<ops::IndexOp>().GetResult();
        const auto sample = weigh.Create<ops::SampleOp>(data, idx).GetResult();
        const auto result = weigh.Create<ops::ArithmeticOp>(weight, sample, ops::eArithmeticFunction::MUL).GetResult();
        weigh.Create<ops::ReturnOp>(std::vector{ result });
    }

    auto add = mod.Create<ops::StencilOp>("add", FunctionType::Get({ fieldT, fieldT }, { scalarT }), 2, false);
    {
        const auto lhs = add.GetRegionArg(0);
        const auto rhs = add.GetRegionArg(1);
        const auto idx = add.Create<ops::IndexOp>().GetResult();
        const auto slhs = add.Create<ops::SampleOp>(lhs, idx).GetResult();
        const auto srhs = add.Create<ops::SampleOp>(rhs, idx).GetResult();
        const auto result = add.Create<ops::ArithmeticOp>(slhs, srhs, ops::eArithmeticFunction::ADD).GetResult();
        add.Create<ops::ReturnOp>(std::vector{ result });
    }

    auto ddx = mod.Create<ops::FuncOp>("ddx", FunctionType::Get({ fieldT, fieldT }, { fieldT }), false);
    {
        const auto c0 = ddx.Create<ops::ConstantOp>(0, indexT).GetResult();
        const auto c1 = ddx.Create<ops::ConstantOp>(1, indexT).GetResult();
        const auto c2 = ddx.Create<ops::ConstantOp>(2, indexT).GetResult();
        const auto c3 = ddx.Create<ops::ConstantOp>(3, indexT).GetResult();
        const auto c4 = ddx.Create<ops::ConstantOp>(4, indexT).GetResult();
        const auto input = ddx.GetRegionArg(0);
        const auto output = ddx.GetRegionArg(1);
        const auto inputx = ddx.Create<ops::DimOp>(input, c0).GetResult();
        const auto slcx = ddx.Create<ops::ArithmeticOp>(inputx, c4, ops::eArithmeticFunction::SUB).GetResult();
        const auto slcy = ddx.Create<ops::DimOp>(input, c1).GetResult();

        const auto slc2m = ddx.Create<ops::ExtractSliceOp>(input, std::vector{ c0, c0 }, std::vector{ slcx, slcy }, std::vector{ c1, c1 }).GetResult();
        const auto slc1m = ddx.Create<ops::ExtractSliceOp>(input, std::vector{ c1, c0 }, std::vector{ slcx, slcy }, std::vector{ c1, c1 }).GetResult();
        const auto slc1p = ddx.Create<ops::ExtractSliceOp>(input, std::vector{ c3, c0 }, std::vector{ slcx, slcy }, std::vector{ c1, c1 }).GetResult();
        const auto slc2p = ddx.Create<ops::ExtractSliceOp>(input, std::vector{ c4, c0 }, std::vector{ slcx, slcy }, std::vector{ c1, c1 }).GetResult();

        const auto c1f = ddx.Create<ops::ConstantOp>(1.0 / 12, scalarT).GetResult();
        const auto cm8f = ddx.Create<ops::ConstantOp>(-8.0 / 12, scalarT).GetResult();
        const auto c8f = ddx.Create<ops::ConstantOp>(8.0 / 12, scalarT).GetResult();
        const auto cm1f = ddx.Create<ops::ConstantOp>(-1.0 / 12, scalarT).GetResult();

        const auto wslc2mOut = ddx.Create<ops::AllocTensorOp>(scalarT, std::vector{ slcx, slcy }).GetResult();
        const auto wslc2m = ddx.Create<ops::ApplyOp>(weigh, std::vector{ c1f, slc2m }, std::vector{ wslc2mOut }, std::vector<Value>{}, std::vector<int64_t>{ 0, 0 }).GetResults()[0];
        const auto wslc1mOut = ddx.Create<ops::AllocTensorOp>(scalarT, std::vector{ slcx, slcy }).GetResult();
        const auto wslc1m = ddx.Create<ops::ApplyOp>(weigh, std::vector{ cm8f, slc1m }, std::vector{ wslc1mOut }, std::vector<Value>{}, std::vector<int64_t>{ 0, 0 }).GetResults()[0];
        const auto wslc1pOut = ddx.Create<ops::AllocTensorOp>(scalarT, std::vector{ slcx, slcy }).GetResult();
        const auto wslc1p = ddx.Create<ops::ApplyOp>(weigh, std::vector{ c8f, slc1p }, std::vector{ wslc1pOut }, std::vector<Value>{}, std::vector<int64_t>{ 0, 0 }).GetResults()[0];
        const auto wslc2pOut = ddx.Create<ops::AllocTensorOp>(scalarT, std::vector{ slcx, slcy }).GetResult();
        const auto wslc2p = ddx.Create<ops::ApplyOp>(weigh, std::vector{ cm1f, slc2p }, std::vector{ wslc2pOut }, std::vector<Value>{}, std::vector<int64_t>{ 0, 0 }).GetResults()[0];

        const auto s0Out = ddx.Create<ops::AllocTensorOp>(scalarT, std::vector{ slcx, slcy }).GetResult();
        const auto s0 = ddx.Create<ops::ApplyOp>(add, std::vector{ wslc2m, wslc1m }, std::vector{ s0Out }, std::vector<Value>{}, std::vector<int64_t>{ 0, 0 }).GetResults()[0];
        const auto s1Out = ddx.Create<ops::AllocTensorOp>(scalarT, std::vector{ slcx, slcy }).GetResult();
        const auto s1 = ddx.Create<ops::ApplyOp>(add, std::vector{ wslc1p, wslc2p }, std::vector{ s1Out }, std::vector<Value>{}, std::vector<int64_t>{ 0, 0 }).GetResults()[0];
        const auto s2Out = ddx.Create<ops::AllocTensorOp>(scalarT, std::vector{ slcx, slcy }).GetResult();
        const auto s2 = ddx.Create<ops::ApplyOp>(add, std::vector{ s0, s1 }, std::vector{ s2Out }, std::vector<Value>{}, std::vector<int64_t>{ 0, 0 }).GetResults()[0];

        const auto rv = ddx.Create<ops::InsertSliceOp>(s2, output, std::vector{ c0, c0 }, std::vector{ slcx, slcy }, std::vector{ c1, c1 }).GetResult();
        ddx.Create<ops::ReturnOp>(std::vector{ rv });
    }

    auto main = mod.Create<ops::FuncOp>("main", FunctionType::Get({ fieldT, fieldT }, { fieldT }));
    {
        const auto c0 = main.Create<ops::ConstantOp>(0, indexT).GetResult();
        const auto c1 = main.Create<ops::ConstantOp>(1, indexT).GetResult();
        const auto c4 = main.Create<ops::ConstantOp>(4, indexT).GetResult();
        const auto input = main.GetRegionArg(0);
        const auto output = main.GetRegionArg(1);
        const auto inputx = main.Create<ops::DimOp>(input, c0).GetResult();
        const auto slcy = main.Create<ops::DimOp>(input, c1).GetResult();

        const auto slcx1 = main.Create<ops::ArithmeticOp>(inputx, c4, ops::eArithmeticFunction::SUB).GetResult();
        const auto out1 = main.Create<ops::AllocTensorOp>(scalarT, std::vector{ slcx1, slcy }).GetResult();
        const auto d = main.Create<ops::CallOp>(ddx, std::vector{ input, out1 }).GetResults()[0];

        const auto slcx2 = main.Create<ops::ArithmeticOp>(slcx1, c4, ops::eArithmeticFunction::SUB).GetResult();
        const auto out2 = main.Create<ops::AllocTensorOp>(scalarT, std::vector{ slcx2, slcy }).GetResult();
        const auto dd = main.Create<ops::CallOp>(ddx, std::vector{ d, out2 }).GetResults()[0];

        auto rv = main.Create<ops::InsertSliceOp>(dd, output, std::vector{ c0, c0 }, std::vector{ slcx2, slcy }, std::vector{ c1, c1 }).GetResult();
        main.Create<ops::ReturnOp>(std::vector<Value>{ rv });
    }

    return mod;
}

TEST_CASE("Optimization #2", "[Program]") {
    constexpr ptrdiff_t sizeX = 20;
    constexpr ptrdiff_t sizeY = 3;
    constexpr ptrdiff_t outputX = sizeX - 8;
    constexpr ptrdiff_t outputY = sizeY;
    std::array<float, sizeX * sizeY> inputBuffer;
    std::array<float, outputX * outputY> outputBuffer;
    std::array<float, outputX * outputY> expectedBuffer;
    StridedMemRefType<float, 2> input{ inputBuffer.data(), inputBuffer.data(), 0, { sizeX, sizeY }, { 1, sizeX } };
    StridedMemRefType<float, 2> output{ outputBuffer.data(), outputBuffer.data(), 0, { outputX, outputY }, { 1, outputX } };
    StridedMemRefType<float, 2> result{ nullptr, nullptr, 0, { 0, 0 }, { 0, 0 } };
    std::ranges::fill(outputBuffer, 0);

    for (ptrdiff_t x = 0; x < sizeX; ++x) {
        const auto xv = float(x);
        inputBuffer[x + 0 * sizeX] = std::exp(0.1 * xv);
        inputBuffer[x + 1 * sizeX] = xv * xv * xv;
        inputBuffer[x + 2 * sizeX] = std::sin(0.2 * xv);
        const auto xo = x - 4;
        if (0 <= xo && xo < outputX) {
            expectedBuffer[xo + 0 * outputX] = 0.01 * std::exp(0.1 * xv);
            expectedBuffer[xo + 1 * outputX] = 6 * xv;
            expectedBuffer[xo + 2 * outputX] = -0.04 * std::sin(0.2 * xv);
        }
    }

    const auto program = CreateModule();
    try {
        RunModule(program, "main", true, input, output, Runner::Result{ result });

        const auto maxDifference = std::inner_product(
            outputBuffer.begin(),
            outputBuffer.end(),
            expectedBuffer.begin(),
            0.0f,
            [](float acc, float v) { return std::max(acc, v); },
            [](float u, float v) { return std::abs(u - v); });
        REQUIRE(maxDifference < 0.005f);
    }
    catch (CompilationError& ex) {
        const auto outPath = std::filesystem::temp_directory_path() / "99_failed_stage.mlir";
        std::ofstream outFile{ outPath, std::ios::trunc };
        outFile << ex.GetModule();
        throw;
    }
}


static ops::ModuleOp CreateAST() {
    auto moduleOp = ops::ModuleOp{};

    auto snSubstract = moduleOp.Create<ops::StencilOp>(
        "subtract",
        FunctionType::Get({ FieldType::Get(Float32, 1),
                            FieldType::Get(Float32, 1) },
                          { Float32 }),
        1);
    auto idx = snSubstract.Create<ops::IndexOp>().GetResult();
    auto lSample = snSubstract.Create<ops::SampleOp>(snSubstract.GetRegionArg(0), idx).GetResult();
    auto rSample = snSubstract.Create<ops::SampleOp>(snSubstract.GetRegionArg(1), idx).GetResult();
    auto sum = snSubstract.Create<ops::ArithmeticOp>(lSample, rSample, ops::eArithmeticFunction::SUB)
                   .GetResult();
    snSubstract.Create<ops::ReturnOp>(std::vector{ sum });


    auto fnMain = moduleOp.Create<ops::FuncOp>(
        "main",
        FunctionType::Get({ FieldType::Get(Float32, 1), FieldType::Get(Float32, 1) }, {}));


    auto input = fnMain.GetRegionArg(0);
    auto output = fnMain.GetRegionArg(1);
    auto czero = fnMain.Create<ops::ConstantOp>(0, IndexType::Get()).GetResult();
    auto cone = fnMain.Create<ops::ConstantOp>(1, IndexType::Get()).GetResult();
    auto size = fnMain.Create<ops::DimOp>(input, czero).GetResult();
    auto dsize = fnMain.Create<ops::ArithmeticOp>(size, cone, ops::eArithmeticFunction::SUB).GetResult();
    auto ddsize = fnMain.Create<ops::ArithmeticOp>(dsize, cone, ops::eArithmeticFunction::SUB).GetResult();


    auto left = fnMain.Create<ops::ExtractSliceOp>(input, std::vector{ czero }, std::vector{ dsize }, std::vector{ cone }).GetResult();
    auto right = fnMain.Create<ops::ExtractSliceOp>(input, std::vector{ cone }, std::vector{ dsize }, std::vector{ cone }).GetResult();
    auto tmp1 = fnMain.Create<ops::AllocTensorOp>(Float32, std::vector{ dsize }).GetResult();

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
    RunModule(program, "main", true, input, output);

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

    REQUIRE(maxDifference < 0.001f);
}