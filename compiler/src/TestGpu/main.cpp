#include <AST/AST.hpp>
#include <AST/LowerToIR.hpp>
#include <AST/Node.hpp>
#include <AST/Types.hpp>
#include <Compiler/Lowering.hpp>
#include <Execution/Execution.hpp>

#include <iostream>
#include <string_view>


std::shared_ptr<ast::Module> CreateDdx() {
    // Kernel logic
    const auto field = std::make_shared<ast::SymbolRef>("field");

    const auto index = std::make_shared<ast::Index>();

    const auto left = std::make_shared<ast::Jump>(index, std::vector<int64_t>{ -1, 0 });
    const auto right = std::make_shared<ast::Jump>(index, std::vector<int64_t>{ 1, 0 });

    const auto sleft = std::make_shared<ast::Sample>(field, left);
    const auto sright = std::make_shared<ast::Sample>(field, right);

    const auto ddx = std::make_shared<ast::Sub>(sright, sleft);
    auto ret = std::make_shared<ast::KernelReturn>(std::vector<std::shared_ptr<ast::Expression>>{ ddx });

    std::vector<ast::Parameter> kernelParams = {
        { "field", types::FieldType{ types::FundamentalType::FLOAT32, 2 } },
    };
    std::vector<types::Type> kernelReturns = { types::FundamentalType::FLOAT32 };
    std::vector<std::shared_ptr<ast::Statement>> kernelBody{ ret };
    auto kernel = std::make_shared<ast::KernelFunc>("ddx",
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

    std::shared_ptr<ast::Module> ast = CreateDdx();
    try {
        mlir::ModuleOp module = LowerToIR(context, *ast);
        ApplyLocationSnapshot(context, module);
        DumpIR(module, "Stencil original");
        ApplyCleanupPasses(context, module);
        DumpIR(module, "Stencil cleaned");

        auto llvmCpuStages = LowerToLLVMGPU(context, module);
        for (auto& stage : llvmCpuStages) {
            DumpIR(stage.second, stage.first);
        }
    }
    catch (std::exception& ex) {
        std::cout << ex.what() << std::endl;
    }
}