#include "AST/AST.hpp"
#include "AST/Node.hpp"
#include "AST/Types.hpp"
#include "AllToLLVMPass.hpp"
#include "Compiler/LowerAST.hpp"
#include "KernelToAffinePass.hpp"
#include "MockPrintPass.hpp"
#include "MockToArithPass.hpp"
#include "llvm/ADT/ArrayRef.h"
#include <MockDialect/MockDialect.hpp>
#include <MockDialect/MockOps.hpp>
#include <iostream>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorOr.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>



std::shared_ptr<ast::Module> CreateProgram() {
    // Kernel logic
    auto a = std::make_shared<ast::SymbolRef>("a");
    auto b = std::make_shared<ast::SymbolRef>("b");
    auto add = std::make_shared<ast::Add>(a, b);
    auto print = std::make_shared<ast::Print>(add);
    auto ret = std::make_shared<ast::KernelReturn>();

    std::vector<ast::Parameter> kernelParams = {
        { "a", types::FundamentalType::FLOAT32 },
        { "b", types::FundamentalType::FLOAT32 },
    };
    std::vector<types::Type> kernelReturns = {};
    std::vector<std::shared_ptr<ast::Statement>> kernelBody{ print, ret };
    auto kernel = std::make_shared<ast::KernelFunc>("kernel_fun",
                                                    kernelParams,
                                                    kernelReturns,
                                                    kernelBody);

    // Main function logic
    std::vector<ast::Parameter> moduleParams = {
        { "a", types::FundamentalType::FLOAT32 },
        { "b", types::FundamentalType::FLOAT32 },
    };

    auto c1 = std::make_shared<ast::SymbolRef>("a");
    auto c2 = std::make_shared<ast::SymbolRef>("b");

    std::vector<std::shared_ptr<ast::Expression>> gridDim = {
        std::make_shared<ast::Constant<int64_t>>(12, types::FundamentalType{ types::FundamentalType::SSIZE }),
        std::make_shared<ast::Constant<int64_t>>(7, types::FundamentalType{ types::FundamentalType::SSIZE })
    };
    std::vector<std::shared_ptr<ast::Expression>> kernelArgs{
        c1,
        c2
    };
    auto kernelLaunch = std::make_shared<ast::KernelLaunch>("kernel_fun",
                                                            gridDim,
                                                            kernelArgs);

    return std::make_shared<ast::Module>(std::vector<std::shared_ptr<ast::Node>>{ kernelLaunch },
                                         std::vector<std::shared_ptr<ast::KernelFunc>>{ kernel },
                                         moduleParams);
}

mlir::LogicalResult LowerToCPU(mlir::MLIRContext& context, mlir::ModuleOp& op) {
    mlir::PassManager passManager(&context);
    passManager.addPass(std::make_unique<MockPrintPass>());
    passManager.addPass(std::make_unique<KernelToAffinePass>());
    return passManager.run(op);
}

mlir::LogicalResult PrepareLoweringToLLVM(mlir::MLIRContext& context, mlir::ModuleOp& op) {
    mlir::PassManager passManager(&context);
    passManager.addPass(std::make_unique<PrepareToLLVMPass>());
    return passManager.run(op);
}

mlir::LogicalResult LowerToLLVM(mlir::MLIRContext& context, mlir::ModuleOp& op) {
    mlir::PassManager passManager(&context);
    passManager.addPass(std::make_unique<AllToLLVMPass>());
    return passManager.run(op);
}

bool ExecuteProgram(mlir::ModuleOp& module, auto&... args) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // Register the translation from MLIR to LLVM IR, which must happen before we
    // can JIT-compile.
    mlir::registerLLVMDialectTranslation(*module.getContext());

    // An optimization pipeline to use within the execution engine.
    constexpr int optLevel = 0;
    constexpr int sizeLevel = 0;
    constexpr auto targetMachine = nullptr;
    auto optPipeline = mlir::makeOptimizingTransformer(optLevel, sizeLevel, targetMachine);

    // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
    // the module.
    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;
    auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
    assert(maybeEngine && "failed to construct an execution engine");
    auto& engine = maybeEngine.get();

    // Invoke the JIT-compiled function.
    std::array opaqueArgs = { static_cast<void*>(&args)... };
    auto invocationResult = engine->invokePacked("main", opaqueArgs);
    if (invocationResult) {
        llvm::errs() << "JIT invocation failed\n";
        return false;
    }

    return true;
}

int main() {
    mlir::MLIRContext context;

    std::shared_ptr<ast::Module> ast = CreateProgram();
    mlir::ModuleOp module = LowerAST(context, *ast);
    std::cout << "Mock IR:\n"
              << std::endl;
    module->dump();

    float a = 3.5;
    float b = 2.6;

    if (LowerToCPU(context, module).succeeded()) {
        std::cout << "\n\nMixed IR:\n"
                  << std::endl;
        module->dump();

        if (PrepareLoweringToLLVM(context, module).succeeded()) {
            std::cout << "\n\nPre-lowered IR:\n"
                      << std::endl;
            module->dump();

            if (LowerToLLVM(context, module).succeeded()) {
                std::cout << "\n\nLLVM IR:\n"
                          << std::endl;
                module->dump();

                if (!ExecuteProgram(module, a, b)) {
                    std::cout << "failed to run!" << std::endl;
                }
            }
        }
    }
    else {
        std::cout << "failed to lower!" << std::endl;
    }
}