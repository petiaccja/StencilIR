#include "AST/AST.hpp"
#include "AST/Node.hpp"
#include "AST/Types.hpp"
#include "Compiler/LowerAST.hpp"
#include "KernelToAffinePass.hpp"
#include "LoweringPasses.hpp"
#include "MockPrintPass.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include <MockDialect/MockDialect.hpp>
#include <MockDialect/MockOps.hpp>
#include <bits/utility.h>
#include <cstddef>
#include <exception>
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
#include <mlir/Dialect/Affine/Passes.h>
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
#include <stdexcept>
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
                                                    kernelBody,
                                                    2);

    // Main function logic
    auto inputA = std::make_shared<ast::SymbolRef>("a");
    auto inputB = std::make_shared<ast::SymbolRef>("b");
    auto output = std::make_shared<ast::SymbolRef>("out");
    auto sizeX = std::make_shared<ast::SymbolRef>("sizeX");
    auto sizeY = std::make_shared<ast::SymbolRef>("sizeY");

    std::vector<std::shared_ptr<ast::Expression>> gridDim = {
        sizeX,
        sizeY,
    };
    std::vector<std::shared_ptr<ast::Expression>> kernelArgs{
        inputA,
        inputB,
    };
    std::vector<std::shared_ptr<ast::Expression>> kernelTargets{
        output,
    };
    auto kernelLaunch = std::make_shared<ast::KernelLaunch>("kernel_fun",
                                                            gridDim,
                                                            kernelArgs,
                                                            kernelTargets);

    // Module
    auto moduleParams = std::vector<ast::Parameter>{
        { "a", types::FundamentalType::FLOAT32 },
        { "b", types::FundamentalType::FLOAT32 },
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

void ThrowIfFailed(mlir::LogicalResult result, std::string msg) {
    if (failed(result)) {
        throw std::runtime_error(std::move(msg));
    }
}

void ApplyLowerToAffine(mlir::MLIRContext& context, mlir::ModuleOp& op) {
    mlir::PassManager passManager(&context);
    passManager.addPass(std::make_unique<KernelToAffinePass>());
    passManager.addPass(std::make_unique<MockPrintPass>());
    ThrowIfFailed(passManager.run(op), "Failed to lower to Affine.");

    passManager.clear();
    passManager.addNestedPass<mlir::func::FuncOp>(mlir::createAffineParallelizePass());
    ThrowIfFailed(passManager.run(op), "Failed to parallelize Affine.");
}

void ApplyLowerToScf(mlir::MLIRContext& context, mlir::ModuleOp& op) {
    mlir::PassManager passManager(&context);
    passManager.addPass(std::make_unique<AffineToScfPass>());
    ThrowIfFailed(passManager.run(op), "Failed to lower to SCF.");
}

void ApplyLowerToCf(mlir::MLIRContext& context, mlir::ModuleOp& op) {
    mlir::PassManager passManager(&context);
    passManager.addPass(std::make_unique<ScfToCfPass>());
    ThrowIfFailed(passManager.run(op), "Failed to lower to ControlFlow.");
}

void ApplyLowerToLLVM(mlir::MLIRContext& context, mlir::ModuleOp& op) {
    mlir::PassManager passManager(&context);
    passManager.addPass(std::make_unique<StdToLLVMPass>());
    ThrowIfFailed(passManager.run(op), "Failed to lower to LLVM IR.");
}

void ApplyCleanupPasses(mlir::MLIRContext& context, mlir::ModuleOp& op) {
    mlir::PassManager passManager(&context);
    passManager.addPass(mlir::createCSEPass());
    passManager.addPass(mlir::createCanonicalizerPass());
    passManager.addPass(mlir::createTopologicalSortPass());
    ThrowIfFailed(passManager.run(op), "Failed to clean up.");
}

void ApplyLocationSnapshot(mlir::MLIRContext& context, mlir::ModuleOp& op) {
    const auto tempPath = std::filesystem::temp_directory_path();
    const auto tempFile = tempPath / "mock.mlir";
    const auto fileName = tempFile.string();

    mlir::PassManager passManager(&context);
    passManager.addPass(mlir::createLocationSnapshotPass({}, fileName));
    ThrowIfFailed(passManager.run(op), "Failed to snapshot locations.");
}

template <class T, size_t Dim>
struct MemRef {
    T* ptr;
    T* alignedPtr;
    ptrdiff_t offset;
    std::array<ptrdiff_t, Dim> shape;
    std::array<ptrdiff_t, Dim> strides;
};

auto ConvertArg(const std::floating_point auto& arg) {
    return std::tuple{ arg };
}

auto ConvertArg(const std::integral auto& arg) {
    return std::tuple{ arg };
}

template <class T, size_t Dim, size_t... Indices>
auto ArrayToTupleHelper(const std::array<T, Dim>& arr, std::index_sequence<Indices...>) {
    return std::make_tuple(arr[Indices]...);
}

template <class T, size_t Dim>
auto ArrayToTuple(const std::array<T, Dim>& arr) {
    return ArrayToTupleHelper(arr, std::make_index_sequence<Dim>());
}

template <class T, size_t Dim>
auto ConvertArg(const MemRef<T, Dim>& arg) {
    return std::tuple_cat(std::tuple{ arg.ptr, arg.alignedPtr, arg.offset },
                          ArrayToTuple(arg.shape),
                          ArrayToTuple(arg.strides));
}

template <class... Args>
auto ConvertArgs(const Args&... args) {
    return std::tuple_cat(ConvertArg(args)...);
}

template <class... Args, size_t... Indices>
auto OpaqueArgsHelper(std::tuple<Args...>& args, std::index_sequence<Indices...>) {
    return std::array{ static_cast<void*>(std::addressof(std::get<Indices>(args)))... };
}

template <class... Args>
auto OpaqueArgs(std::tuple<Args...>& args) {
    return OpaqueArgsHelper(args, std::make_index_sequence<sizeof...(Args)>());
}


void ExecuteProgram(mlir::ModuleOp& module, auto&... args) {
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
    if (!maybeEngine) {
        throw std::runtime_error("failed to construct an execution engine");
    }
    auto& engine = maybeEngine.get();

    // Invoke the JIT-compiled function.
    auto convertedArgs = ConvertArgs(args...);
    std::array opaqueArgs = OpaqueArgs(convertedArgs);
    auto invocationResult = engine->invokePacked("main", opaqueArgs);
    if (invocationResult) {
        throw std::runtime_error("Invoking JIT-ed function failed.");
    }
}

int main() {
    mlir::MLIRContext context;

    std::shared_ptr<ast::Module> ast = CreateProgram();
    try {
        // Generate module from AST
        mlir::ModuleOp module = LowerAST(context, *ast);
        ApplyLocationSnapshot(context, module);
        std::cout << "Mock IR:\n\n";
        module.dump();

        // Lower module to LLVM IR
        ApplyLowerToAffine(context, module);
        ApplyCleanupPasses(context, module);
        std::cout << "\n\nAffine IR:\n\n";
        module.dump();

        ApplyLowerToScf(context, module);
        ApplyCleanupPasses(context, module);
        std::cout << "\n\nSFC IR:\n\n";
        module.dump();

        ApplyLowerToCf(context, module);
        ApplyCleanupPasses(context, module);
        std::cout << "\n\nControlFLow IR:\n\n";
        module.dump();

        ApplyLowerToLLVM(context, module);
        ApplyCleanupPasses(context, module);
        std::cout << "\n\nLLVM IR:\n\n";
        module.dump();


        // Execute JIT-ed module.
        float a = 3.5;
        float b = 2.6;
        constexpr ptrdiff_t inputSizeX = 9;
        constexpr ptrdiff_t inputSizeY = 7;
        constexpr ptrdiff_t domainSizeX = inputSizeX - 2;
        constexpr ptrdiff_t domainSizeY = inputSizeY - 2;
        std::array<float, inputSizeX * inputSizeY> outputBuffer;
        MemRef<float, 2> output{ outputBuffer.data(), outputBuffer.data(), inputSizeX + 1, { domainSizeX, domainSizeY }, { 1, inputSizeX } };
        std::ranges::fill(outputBuffer, 0);

        ExecuteProgram(module, a, b, output, domainSizeX, domainSizeY);

        for (size_t y = 0; y < inputSizeY; ++y) {
            for (size_t x = 0; x < inputSizeX; ++x) {
                std::cout << outputBuffer[y * inputSizeX + x] << " ";
            }
            std::cout << std::endl;
        }
    }
    catch (std::exception& ex) {
        std::cout << ex.what() << std::endl;
    }
}