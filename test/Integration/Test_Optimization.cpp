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
    // Stencils
    auto subtract = ast::stencil("subtract",
                                 { { "a", ast::FieldType::Get(ast::FloatType::Get(32), 1) },
                                   { "b", ast::FieldType::Get(ast::FloatType::Get(32), 1) } },
                                 { ast::FloatType::Get(32) },
                                 { ast::return_({ ast::sub(
                                     ast::sample(ast::symref("a"), ast::index()),
                                     ast::sample(ast::symref("b"), ast::index())) }) },
                                 1);

    // Main function logic
    auto input = ast::symref("input");
    auto output = ast::symref("output");

    auto czero = ast::constant(0, ast::IndexType::Get());
    auto cone = ast::constant(1, ast::IndexType::Get());
    auto size = ast::dim(input, ast::constant(0, ast::IndexType::Get()));
    auto dsize = ast::sub(size, ast::constant(1, ast::IndexType::Get()));
    auto ddsize = ast::sub(dsize, ast::constant(1, ast::IndexType::Get()));

    auto left = ast::extract_slice(input, { czero }, { dsize }, { cone });
    auto right = ast::extract_slice(input, { cone }, { dsize }, { cone });

    auto tmp1 = ast::alloc_tensor(ast::FloatType::Get(32), { dsize });
    auto d = ast::apply(subtract->name,
                        { right, left },
                        { tmp1 },
                        std::vector<int64_t>{ 0, 0 });
    auto [dass, dvar] = std::tuple{ ast::assign({ "dvar" }, d), ast::symref("dvar") };

    auto dleft = ast::extract_slice(dvar, { czero }, { ddsize }, { cone });
    auto dright = ast::extract_slice(dvar, { cone }, { ddsize }, { cone });
    auto dd = ast::apply(subtract->name,
                         { dright, dleft },
                         { output },
                         std::vector<int64_t>{ 0, 0 });

    auto main = ast::function("main",
                              {
                                  { "input", ast::FieldType::Get(ast::FloatType::Get(32), 1) },
                                  { "output", ast::FieldType::Get(ast::FloatType::Get(32), 1) },
                              },
                              {},
                              { dass, dd, ast::return_() });


    return ast::module_({ main },
                        { subtract });
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
        inputBuffer[x] = (x * x * x) * 0.1f;
    }

    const auto program = CreateAST();
    const auto stages = RunAST(*program, "main", true, input, output);

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