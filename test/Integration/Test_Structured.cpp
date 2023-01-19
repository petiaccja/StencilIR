#include "Utility/RunAST.hpp"

#include <AST/Building.hpp>
#include <AST/ConvertASTToIR.hpp>
#include <Compiler/Pipelines.hpp>
#include <Execution/Execution.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string_view>

#include <catch2/catch.hpp>


static std::shared_ptr<ast::Module> CreateAST() {
    // Kernel logicExecute
    auto field = ast::symref("field");

    std::array samples = {
        ast::mul(ast::constant(4.0f), ast::sample(field, ast::index())),
        ast::mul(ast::constant(-1.0f), ast::sample(field, ast::jump(ast::index(), { 0, 1 }))),
        ast::mul(ast::constant(-1.0f), ast::sample(field, ast::jump(ast::index(), { 1, 0 }))),
        ast::mul(ast::constant(-1.0f), ast::sample(field, ast::jump(ast::index(), { 0, -1 }))),
        ast::mul(ast::constant(-1.0f), ast::sample(field, ast::jump(ast::index(), { -1, 0 }))),
    };

    auto sum = ast::add(samples[0], ast::add(samples[1], ast::add(samples[2], ast::add(samples[3], samples[4]))));
    auto ret = ast::return_({ sum });

    auto laplacian = ast::stencil("laplacian",
                                  { { "field", ast::FieldType::Get(ast::FloatType::Get(32), 2) } },
                                  { ast::FloatType::Get(32) },
                                  { ret },
                                  2);

    // Main function logic
    auto input = ast::symref("input");
    auto output = ast::symref("out");

    auto apply = ast::apply(laplacian->name,
                            { input },
                            { output },
                            { 1, 1 });

    auto main = ast::function("main",
                              {
                                  { "input", ast::FieldType::Get(ast::FloatType::Get(32), 2) },
                                  { "out", ast::FieldType::Get(ast::FloatType::Get(32), 2) },
                              },
                              {},
                              { apply, ast::return_() });


    return ast::module_({ main },
                        { laplacian });
}

TEST_CASE("Structured", "[Program]") {
    constexpr ptrdiff_t inputSizeX = 9;
    constexpr ptrdiff_t inputSizeY = 7;
    constexpr ptrdiff_t outputSizeX = inputSizeX - 2;
    constexpr ptrdiff_t outputSizeY = inputSizeY - 2;
    std::array<float, inputSizeX * inputSizeY> inputBuffer;
    std::array<float, outputSizeX * outputSizeY> outputBuffer;
    StridedMemRefType<float, 2> input{ inputBuffer.data(), inputBuffer.data(), 0, { inputSizeX, inputSizeY }, { 1, inputSizeX } };
    StridedMemRefType<float, 2> output{ outputBuffer.data(), outputBuffer.data(), 0, { outputSizeX, outputSizeY }, { 1, outputSizeX } };
    std::ranges::fill(outputBuffer, 0);

    for (size_t y = 0; y < inputSizeY; ++y) {
        for (size_t x = 0; x < inputSizeX; ++x) {
            inputBuffer[y * inputSizeX + x] = (x * x * x + y * y) * 0.1f;
        }
    }

    const auto program = CreateAST();
    const auto stages = RunAST(*program, "main", input, output);

    const std::array<float, outputSizeX* outputSizeY> expectedBuffer = {
        -0.8f, -1.4f, -2.0f, -2.6f, -3.2f, -3.8f, -4.4f,
        -0.8f, -1.4f, -2.0f, -2.6f, -3.2f, -3.8f, -4.4f,
        -0.8f, -1.4f, -2.0f, -2.6f, -3.2f, -3.8f, -4.4f,
        -0.8f, -1.4f, -2.0f, -2.6f, -3.2f, -3.8f, -4.4f,
        -0.8f, -1.4f, -2.0f, -2.6f, -3.2f, -3.8f, -4.4f
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