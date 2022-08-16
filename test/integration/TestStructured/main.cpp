#include <AST/AST.hpp>
#include <AST/Build.hpp>
#include <AST/LowerToIR.hpp>
#include <AST/Node.hpp>
#include <AST/Types.hpp>
#include <Compiler/Lowering.hpp>
#include <Execution/Execution.hpp>

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
                                   { output });

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
    constexpr ptrdiff_t domainSizeX = inputSizeX - 2;
    constexpr ptrdiff_t domainSizeY = inputSizeY - 2;
    std::array<float, inputSizeX * inputSizeY> outputBuffer;
    std::array<float, inputSizeX * inputSizeY> inputBuffer;
    MemRef<float, 2> output{ outputBuffer.data(), outputBuffer.data(), inputSizeX + 1, { domainSizeX, domainSizeY }, { 1, inputSizeX } };
    MemRef<float, 2> input{ inputBuffer.data(), inputBuffer.data(), inputSizeX + 1, { domainSizeX, domainSizeY }, { 1, inputSizeX } };
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
    for (size_t y = 0; y < inputSizeY; ++y) {
        for (size_t x = 0; x < inputSizeX; ++x) {
            std::cout << std::setprecision(4) << outputBuffer[y * inputSizeX + x] << "\t";
        }
        std::cout << std::endl;
    }
}

void DumpIR(mlir::ModuleOp ir, std::string_view name = {}) {
    if (!name.empty()) {
        std::cout << name << ":\n"
                  << std::endl;
    }
    ir->dump();
    std::cout << "\n"
              << std::endl;
}


int main() {
    mlir::MLIRContext context;

    std::shared_ptr<ast::Module> ast = CreateLaplacian();
    try {
        mlir::ModuleOp module = LowerToIR(context, *ast);
        ApplyLocationSnapshot(context, module);
        DumpIR(module, "Stencil original");
        ApplyCleanupPasses(context, module);
        DumpIR(module, "Stencil cleaned");

        auto llvmCpuStages = LowerToLLVMCPU(context, module);
        for (auto& stage : llvmCpuStages) {
            DumpIR(stage.second, stage.first);
        }

        constexpr int optLevel = 3;
        JitRunner jitRunner{ llvmCpuStages.back().second, optLevel };

        std::cout << "LLVM IR:\n"
                  << jitRunner.LLVMIR() << "\n"
                  << std::endl;

        RunLaplacian(jitRunner);
    }
    catch (std::exception& ex) {
        std::cout << ex.what() << std::endl;
    }
}