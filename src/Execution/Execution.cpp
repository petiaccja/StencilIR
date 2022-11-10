#include "Execution.hpp"

#include "DynamicLinking.hpp"

#include <llvm/Support/TargetSelect.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>


Runner::Runner(mlir::ModuleOp& llvmIr, int optLevel) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::registerLLVMDialectTranslation(*llvmIr.getContext());

    constexpr int sizeLevel = 0;
    constexpr auto targetMachine = nullptr;
    auto optPipeline = mlir::makeOptimizingTransformer(optLevel, sizeLevel, targetMachine);


    const auto runnerUtilsLibPath = GetModulePath(R"(.*mlir_c_runner_utils.*)");
    if (!runnerUtilsLibPath) {
        throw std::runtime_error("Could not find MLIR runner utilities shared library.");
    }
    const auto runnerUtilsLibPathStr = runnerUtilsLibPath->string();
    std::vector<mlir::StringRef> sharedLibPaths = {
        runnerUtilsLibPathStr
    };

    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;
    engineOptions.enableGDBNotificationListener = true;
    engineOptions.enablePerfNotificationListener = true;
    engineOptions.sharedLibPaths = sharedLibPaths;
    engineOptions.jitCodeGenOptLevel = optLevel == 0   ? llvm::CodeGenOpt::None
                                       : optLevel == 1 ? llvm::CodeGenOpt::Less
                                       : optLevel == 2 ? llvm::CodeGenOpt::Default
                                                       : llvm::CodeGenOpt::Aggressive;
    auto maybeEngine = mlir::ExecutionEngine::create(llvmIr, engineOptions);

    if (!maybeEngine) {
        llvm::Error error = maybeEngine.takeError();
        std::string str;
        llvm::raw_string_ostream os(str);
        llvm::handleAllErrors(std::move(error), [&](const llvm::ErrorInfoBase& error) {
            error.log(os);
        });
        throw std::runtime_error("Failed to construct an execution engine: " + str);
    }

    m_engine = std::move(maybeEngine.get());

    // translate to LLVM IR for testing
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(llvmIr, llvmContext);
    if (!llvmModule) {
        throw std::runtime_error("failed to generate LLVM IR");
    }
    mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());
    if (auto err = optPipeline(llvmModule.get())) {
        throw std::runtime_error("failed to optimize LLVM IR");
    }
    llvm::raw_string_ostream ss{ m_llvmIrDump };
    ss << *llvmModule;
}


void Runner::Invoke(std::string_view name, std::span<void*> args) const {
    llvm::Error error = m_engine->invokePacked(name, { args.data(), args.size() });
    if (error) {
        std::string message;
        llvm::raw_string_ostream os(message);
        error = llvm::handleErrors(
            std::move(error),
            [&](llvm::ErrorInfoBase& err) { err.log(os); });
        if (error) {
            throw std::runtime_error("Error while handling errors. Seriously, what the fuck is this?");
        }
        throw std::runtime_error("Invoking JIT-ed function failed: " + message);
    }
}