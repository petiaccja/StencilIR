#include <AST/AST.hpp>
#include <AST/Build.hpp>
#include <AST/LowerToIR.hpp>
#include <AST/Node.hpp>
#include <AST/Types.hpp>
#include <Compiler/Lowering.hpp>
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

    auto kernel = ast::stencil("laplacian",
                               { { "field", types::FieldType{ types::FundamentalType::FLOAT32, 2 } } },
                               { types::FundamentalType::FLOAT32 },
                               { ret },
                               2);

    // Main function logic
    auto input = ast::symref("input");
    auto output = ast::symref("out");

    auto applyStencil = ast::apply(kernel->name,
                                   { input },
                                   { output },
                                   { 1, 1 });

    // Module
    auto moduleBody = std::vector<std::shared_ptr<ast::Node>>{ applyStencil };
    auto moduleKernels = std::vector<std::shared_ptr<ast::Stencil>>{ kernel };

    return ast::module_({ applyStencil },
                        { kernel },
                        {
                            { "input", types::FieldType{ types::FundamentalType::FLOAT32, 2 } },
                            { "out", types::FieldType{ types::FundamentalType::FLOAT32, 2 } },
                        });
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

std::string StringizeIR(mlir::ModuleOp ir) {
    std::string s;
    llvm::raw_string_ostream ss{ s };
    ir->print(ss, mlir::OpPrintingFlags{}.elideLargeElementsAttrs());
    return s;
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
        mlir::ModuleOp module = LowerToIR(context, *ast);
        ApplyLocationSnapshot(context, module);
        DumpIR(StringizeIR(module), "Stencil original");
        ApplyCleanupPasses(context, module);
        DumpIR(StringizeIR(module), "Stencil cleaned");

        auto llvmCpuStages = LowerToLLVMCPU(context, module);
        for (auto& stage : llvmCpuStages) {
            DumpIR(StringizeIR(stage.second), stage.first);
        }

        constexpr int optLevel = 3;
        JitRunner jitRunner{ llvmCpuStages.back().second, optLevel };

        DumpIR(jitRunner.LLVMIR(), "LLVM IR");

        RunLaplacian(jitRunner);
    }
    catch (std::exception& ex) {
        std::cout << ex.what() << std::endl;
    }
}