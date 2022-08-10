#include "AST/AST.hpp"
#include "AST/Node.hpp"
#include "AST/Types.hpp"
#include "AllToLLVMPass.hpp"
#include "Compiler/LowerAST.hpp"
#include "KernelToAffinePass.hpp"
#include "MockPrintPass.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include <MockDialect/MockDialect.hpp>
#include <MockDialect/MockOps.hpp>
#include <bits/utility.h>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <iterator>
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
#include <mlir/Transforms/Passes.h>
#include <type_traits>


std::shared_ptr<ast::Module> CreateProgram() {
    // Kernel logic
    auto a = std::make_shared<ast::SymbolRef>("a");
    auto b = std::make_shared<ast::SymbolRef>("b");
    auto add = std::make_shared<ast::Add>(a, b);
    auto ret = std::make_shared<ast::KernelReturn>(std::vector<std::shared_ptr<ast::Expression>>{ add });

    std::vector<ast::Parameter> kernelParams = {
        { "a", types::FundamentalType::FLOAT32 },
        { "b", types::FundamentalType::FLOAT32 },
    };
    std::vector<types::Type> kernelReturns = { types::FundamentalType::FLOAT32 };
    std::vector<std::shared_ptr<ast::Statement>> kernelBody{ ret };
    auto kernel = std::make_shared<ast::KernelFunc>("kernel_fun",
                                                    kernelParams,
                                                    kernelReturns,
                                                    kernelBody);

    // Main function logic
    auto inputA = std::make_shared<ast::SymbolRef>("a");
    auto inputB = std::make_shared<ast::SymbolRef>("b");
    auto output = std::make_shared<ast::SymbolRef>("out");
    auto sizeX = std::make_shared<ast::SymbolRef>("sizeX");
    auto sizeY = std::make_shared<ast::SymbolRef>("sizeY");

    auto shape = std::vector<std::shared_ptr<ast::Expression>>{
        sizeX,
        sizeY,
    };
    auto strides = std::vector<std::shared_ptr<ast::Expression>>{
        std::make_shared<ast::Constant<int64_t>>(1, types::FundamentalType::SSIZE),
        sizeX,
    };
    auto reshapedOutput = std::make_shared<ast::ReshapeField>(output, shape, strides);

    std::vector<std::shared_ptr<ast::Expression>> gridDim = {
        sizeX,
        sizeY,
    };
    std::vector<std::shared_ptr<ast::Expression>> kernelArgs{
        inputA,
        inputB,
    };
    std::vector<std::shared_ptr<ast::Expression>> kernelTargets{
        reshapedOutput,
    };
    auto kernelLaunch = std::make_shared<ast::KernelLaunch>("kernel_fun",
                                                            gridDim,
                                                            kernelArgs,
                                                            kernelTargets);

    // Module
    auto moduleParams = std::vector<ast::Parameter>{
        { "a", types::FundamentalType::FLOAT32 },
        { "b", types::FundamentalType::FLOAT32 },
        { "out", types::FieldType{ types::FundamentalType::FLOAT32 } },
        { "sizeX", types::FundamentalType::SSIZE },
        { "sizeY", types::FundamentalType::SSIZE },
    };

    auto moduleBody = std::vector<std::shared_ptr<ast::Node>>{ kernelLaunch };
    auto moduleKernels = std::vector<std::shared_ptr<ast::KernelFunc>>{ kernel };

    return std::make_shared<ast::Module>(moduleBody,
                                         moduleKernels,
                                         moduleParams);
}

mlir::LogicalResult LowerToCPU(mlir::MLIRContext& context, mlir::ModuleOp& op) {
    mlir::PassManager passManager(&context);
    passManager.addPass(std::make_unique<KernelToAffinePass>());
    auto p1 = passManager.run(op);
    if (failed(p1)) {
        return p1;
    }
    mlir::PassManager passManager2(&context);
    passManager2.addPass(std::make_unique<MockPrintPass>());
    return passManager2.run(op);
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

mlir::LogicalResult TidyModule(mlir::MLIRContext& context, mlir::ModuleOp& op) {
    mlir::PassManager passManager(&context);
    passManager.addPass(mlir::createCSEPass());
    passManager.addPass(mlir::createCanonicalizerPass());
    passManager.addPass(mlir::createTopologicalSortPass());
    return passManager.run(op);
}

mlir::LogicalResult SnapshotIR(mlir::MLIRContext& context, mlir::ModuleOp& op) {
    const auto tempPath = std::filesystem::temp_directory_path();
    const auto tempFile = tempPath / "mock.mlir";
    const auto fileName = tempFile.string();

    mlir::PassManager passManager(&context);
    passManager.addPass(mlir::createLocationSnapshotPass({}, fileName));
    return passManager.run(op);
}

auto ConvertArg(std::floating_point auto& arg) {
    return std::tuple{ arg };
}

auto ConvertArg(std::integral auto& arg) {
    return std::tuple{ arg };
}

auto ConvertArg(std::ranges::contiguous_range auto& arg) {
    auto* ptr = std::addressof(*begin(arg));
    size_t size = std::distance(begin(arg), end(arg));
    return std::tuple{ ptr, ptr, size_t(0), size, size_t(1) };
}

template <class... Args>
auto ConvertArgs(Args&... args) {
    return std::tuple_cat(ConvertArg(args)...);
}

template <size_t... Indices, class... Args>
auto OpaqueArgsHelper(std::index_sequence<Indices...>, std::tuple<Args...>& args) {
    return std::array{ static_cast<void*>(std::addressof(std::get<Indices>(args)))... };
}

template <class... Args>
auto OpaqueArgs(std::tuple<Args...>& args) {
    return OpaqueArgsHelper(std::make_index_sequence<sizeof...(Args)>(), args);
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
    auto convertedArgs = ConvertArgs(args...);
    std::array opaqueArgs = OpaqueArgs(convertedArgs);
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
    if (failed(SnapshotIR(context, module))) {
        std::cout << "Snapshotting failed\n"
                  << std::endl;
    }
    std::cout << "Mock IR:\n"
              << std::endl;
    module->dump();

    float a = 3.5;
    float b = 2.6;
    constexpr ptrdiff_t sizeX = 13;
    constexpr ptrdiff_t sizeY = 9;
    std::array<float, sizeX * sizeY * 25> out;
    std::ranges::fill(out, 0);

    if (LowerToCPU(context, module).succeeded() && TidyModule(context, module).succeeded()) {
        std::cout << "\n\nMixed IR:\n"
                  << std::endl;
        module->dump();

        if (PrepareLoweringToLLVM(context, module).succeeded() && TidyModule(context, module).succeeded()) {
            std::cout << "\n\nPre-lowered IR:\n"
                      << std::endl;
            module->dump();

            if (LowerToLLVM(context, module).succeeded() && TidyModule(context, module).succeeded()) {
                std::cout << "\n\nLLVM IR:\n"
                          << std::endl;
                module->dump();

                if (ExecuteProgram(module, a, b, out, sizeX, sizeY)) {
                    for (size_t y = 0; y < sizeY; ++y) {
                        for (size_t x = 0; x < sizeX; ++x) {
                            std::cout << out[y * sizeX + x] << " ";
                        }
                        std::cout << std::endl;
                    }
                }
                else {
                    std::cout << "failed to run!" << std::endl;
                }
            }
        }
    }
    else {
        std::cout << "failed to lower!" << std::endl;
    }
}