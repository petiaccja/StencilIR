#include <AST/ASTBuilding.hpp>
#include <AST/ConvertASTToIR.hpp>
#include <Compiler/Pipelines.hpp>
#include <Execution/Execution.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string_view>


std::shared_ptr<ast::Module> CreateLaplacian() {
    // Kernel logic
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
                                  { { "field", types::FieldType{ types::FundamentalType::FLOAT32, 2 } } },
                                  { types::FundamentalType::FLOAT32 },
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
                                  { "input", types::FieldType{ types::FundamentalType::FLOAT32, 2 } },
                                  { "out", types::FieldType{ types::FundamentalType::FLOAT32, 2 } },
                              },
                              {},
                              { apply, ast::return_() });


    return ast::module_({ main },
                        { laplacian });
}


void RunLaplacian(JitRunner& runner) {
    constexpr ptrdiff_t inputSizeX = 9;
    constexpr ptrdiff_t inputSizeY = 7;
    constexpr ptrdiff_t outputSizeX = inputSizeX - 2;
    constexpr ptrdiff_t outputSizeY = inputSizeY - 2;
    std::array<float, inputSizeX * inputSizeY> inputBuffer;
    std::array<float, outputSizeX * outputSizeY> outputBuffer;
    MemRef<float, 2> input{ inputBuffer.data(), inputBuffer.data(), 0, { inputSizeX, inputSizeY }, { 1, inputSizeX } };
    MemRef<float, 2> output{ outputBuffer.data(), outputBuffer.data(), 0, { outputSizeX, outputSizeY }, { 1, outputSizeX } };
    std::ranges::fill(outputBuffer, 0);

    for (size_t y = 0; y < inputSizeY; ++y) {
        for (size_t x = 0; x < inputSizeX; ++x) {
            inputBuffer[y * inputSizeX + x] = (x * x * x + y * y) * 0.1f;
        }
    }

    runner.InvokeFunction("main", input, output);

    std::cout << "Input:" << std::endl;
    for (size_t y = 0; y < inputSizeY; ++y) {
        for (size_t x = 0; x < inputSizeX; ++x) {
            std::cout << std::setprecision(4) << inputBuffer[y * inputSizeX + x] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << "\nOutput:" << std::endl;
    for (size_t y = 0; y < outputSizeY; ++y) {
        for (size_t x = 0; x < outputSizeX; ++x) {
            std::cout << std::setprecision(4) << outputBuffer[y * outputSizeX + x] << "\t";
        }
        std::cout << std::endl;
    }
}

void DumpIR(std::string_view ir, std::string_view name) {
    assert(!name.empty());
    std::string fname;
    std::transform(name.begin(), name.end(), std::back_inserter(fname), [](char c) {
        if (std::ispunct(c) || std::isspace(c)) {
            c = '_';
        }
        return std::tolower(c);
    });
    auto tempfile = std::filesystem::temp_directory_path() / (fname + ".mlir");
    std::ofstream of(tempfile, std::ios::trunc);
    of << ir;
}


int main() {
    mlir::MLIRContext context;

    std::shared_ptr<ast::Module> ast = CreateLaplacian();
    try {
        mlir::ModuleOp module = ConvertASTToIR(context, *ast);

        Compiler compiler{ TargetCPUPipeline(context) };
        std::vector<StageResult> stageResults;

        mlir::ModuleOp compiled = compiler.Run(module, stageResults);
        for (auto& stageResult : stageResults) {
            DumpIR(stageResult.ir, stageResult.name);
        }

        constexpr int optLevel = 3;
        JitRunner jitRunner{ compiled, optLevel };
        RunLaplacian(jitRunner);

        DumpIR(jitRunner.LLVMIR(), "LLVM IR");
    }
    catch (std::exception& ex) {
        std::cout << ex.what() << std::endl;
    }
}