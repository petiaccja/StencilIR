#include "Execution.hpp"

#include <AST/AST.hpp>
#include <AST/LowerToIR.hpp>
#include <AST/Node.hpp>
#include <AST/Types.hpp>
#include <Compiler/Lowering.hpp>

#include <iostream>
#include <string_view>


std::shared_ptr<ast::Module> CreateLaplacian() {
    // Kernel logic
    auto field = std::make_shared<ast::SymbolRef>("field");

    auto index = std::make_shared<ast::Index>();

    std::array<std::shared_ptr<ast::Expression>, 5> offsetIndices = {
        index,
        std::make_shared<ast::Offset>(index, std::vector<int64_t>{ 0, 1 }),
        std::make_shared<ast::Offset>(index, std::vector<int64_t>{ 1, 0 }),
        std::make_shared<ast::Offset>(index, std::vector<int64_t>{ 0, -1 }),
        std::make_shared<ast::Offset>(index, std::vector<int64_t>{ -1, 0 }),
    };
    std::array<std::shared_ptr<ast::Expression>, 5> samples = {
        std::make_shared<ast::Sample>(field, offsetIndices[0]),
        std::make_shared<ast::Sample>(field, offsetIndices[1]),
        std::make_shared<ast::Sample>(field, offsetIndices[2]),
        std::make_shared<ast::Sample>(field, offsetIndices[3]),
        std::make_shared<ast::Sample>(field, offsetIndices[4]),
    };
    std::array<std::shared_ptr<ast::Expression>, 5> weights = {
        std::make_shared<ast::Constant<float>>(4.0f),
        std::make_shared<ast::Constant<float>>(-1.0f),
        std::make_shared<ast::Constant<float>>(-1.0f),
        std::make_shared<ast::Constant<float>>(-1.0f),
        std::make_shared<ast::Constant<float>>(-1.0f),
    };
    std::array<std::shared_ptr<ast::Expression>, 5> products = {
        std::make_shared<ast::Mul>(samples[0], weights[0]),
        std::make_shared<ast::Mul>(samples[1], weights[1]),
        std::make_shared<ast::Mul>(samples[2], weights[2]),
        std::make_shared<ast::Mul>(samples[3], weights[3]),
        std::make_shared<ast::Mul>(samples[4], weights[4]),
    };

    auto sum = std::make_shared<ast::Add>(products[0], products[1]);
    sum = std::make_shared<ast::Add>(sum, products[2]);
    sum = std::make_shared<ast::Add>(sum, products[3]);
    sum = std::make_shared<ast::Add>(sum, products[4]);

    auto ret = std::make_shared<ast::KernelReturn>(std::vector<std::shared_ptr<ast::Expression>>{ sum });

    std::vector<ast::Parameter> kernelParams = {
        { "field", types::FieldType{ types::FundamentalType::FLOAT32, 2 } },
    };
    std::vector<types::Type> kernelReturns = { types::FundamentalType::FLOAT32 };
    std::vector<std::shared_ptr<ast::Statement>> kernelBody{ ret };
    auto kernel = std::make_shared<ast::KernelFunc>("laplacian",
                                                    kernelParams,
                                                    kernelReturns,
                                                    kernelBody,
                                                    2);

    // Main function logic
    auto inputField = std::make_shared<ast::SymbolRef>("input");
    auto output = std::make_shared<ast::SymbolRef>("out");
    auto sizeX = std::make_shared<ast::SymbolRef>("sizeX");
    auto sizeY = std::make_shared<ast::SymbolRef>("sizeY");

    std::vector<std::shared_ptr<ast::Expression>> gridDim = {
        sizeX,
        sizeY,
    };
    std::vector<std::shared_ptr<ast::Expression>> kernelArgs{
        inputField,
    };
    std::vector<std::shared_ptr<ast::Expression>> kernelTargets{
        output,
    };
    auto kernelLaunch = std::make_shared<ast::KernelLaunch>(kernel->name,
                                                            gridDim,
                                                            kernelArgs,
                                                            kernelTargets);

    // Module
    auto moduleParams = std::vector<ast::Parameter>{
        { "input", types::FieldType{ types::FundamentalType::FLOAT32, 2 } },
        { "out", types::FieldType{ types::FundamentalType::FLOAT32, 2 } },
        { "sizeX", types::FundamentalType::SSIZE },
        { "sizeY", types::FundamentalType::SSIZE },
    };

    auto moduleBody = std::vector<std::shared_ptr<ast::Node>>{ kernelLaunch };
    auto moduleKernels = std::vector<std::shared_ptr<ast::KernelFunc>>{ kernel };

    return std::make_shared<ast::Module>(moduleBody,
                                         moduleKernels,
                                         moduleParams);
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

    runner.InvokeFunction("main", input, output, domainSizeX, domainSizeY);

    std::cout << "Input:" << std::endl;
    for (size_t y = 0; y < inputSizeY; ++y) {
        for (size_t x = 0; x < inputSizeX; ++x) {
            std::cout << inputBuffer[y * inputSizeX + x] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << "\nOutput:" << std::endl;
    for (size_t y = 0; y < inputSizeY; ++y) {
        for (size_t x = 0; x < inputSizeX; ++x) {
            std::cout << outputBuffer[y * inputSizeX + x] << "\t";
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

        JitRunner jitRunner{ llvmCpuStages.back().second };
        RunLaplacian(jitRunner);
    }
    catch (std::exception& ex) {
        std::cout << ex.what() << std::endl;
    }
}