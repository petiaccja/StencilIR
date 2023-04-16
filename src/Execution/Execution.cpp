#include "Execution.hpp"

#include "DynamicLinking.hpp"

#include <Diagnostics/Exception.hpp>

#include <llvm/IR/LegacyPassManager.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/ExecutionEngine/CRunnerUtils.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>


namespace sir {

Runner::Runner(mlir::ModuleOp& llvmIr, int optLevel) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::registerLLVMDialectTranslation(*llvmIr.getContext());

    constexpr int sizeLevel = 0;
    constexpr auto targetMachine = nullptr;
    auto optPipeline = mlir::makeOptimizingTransformer(optLevel, sizeLevel, targetMachine);

    // This is needed on Windows so that the MLIR runner utils DLL is actually linked against.
    [[maybe_unused]] volatile auto forceRunnerUtils = &rtclock;
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

    // Translate to LLVM module for additional work.
    auto llvmContext = std::make_unique<llvm::LLVMContext>();
    if (!llvmContext) {
        throw std::runtime_error("failed to create LLVM context");
    }
    auto llvmModule = mlir::translateModuleToLLVMIR(llvmIr, *llvmContext);
    if (!llvmModule) {
        throw std::runtime_error("failed to generate LLVM IR");
    }
    mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

    if (auto err = optPipeline(llvmModule.get())) {
        throw std::runtime_error("failed to optimize LLVM IR");
    }

    m_llvmContext = std::move(llvmContext);
    m_llvmModule = std::move(llvmModule);
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

llvm::LLVMContext& Runner::GetContext() const {
    return m_llvmModule->getContext();
}

const llvm::DataLayout& Runner::GetDataLayout() const {
    return m_llvmModule->getDataLayout();
}

std::string Runner::GetLLVMIR() const {
    std::string printedLLVMIR;
    llvm::raw_string_ostream ss{ printedLLVMIR };
    ss << *m_llvmModule;
    return ss.str();
}

std::vector<char> Runner::GetObjectFile() const {
    auto targetTriple = m_llvmModule->getTargetTriple();

    std::string errorMsg;
    auto target = llvm::TargetRegistry::lookupTarget(targetTriple, errorMsg);

    // Print an error and exit if we couldn't find the requested target.
    // This generally occurs if we've forgotten to initialise the
    // TargetRegistry or we have a bogus target triple.
    if (!target) {
        throw Exception(errorMsg);
    }

    auto cpu = "generic";
    auto features = "";

    llvm::TargetOptions opt;
    auto relocModel = llvm::Optional<llvm::Reloc::Model>();
    auto targetMachine = target->createTargetMachine(targetTriple, cpu, features, opt, relocModel);

    llvm::SmallVector<char, 128> buffer;
    llvm::raw_svector_ostream os(buffer);

    llvm::legacy::PassManager pass;
    auto FileType = llvm::CGFT_ObjectFile;

    if (targetMachine->addPassesToEmitFile(pass, os, nullptr, FileType)) {
        throw Exception("TargetMachine can't emit a file of this type");
    }

    pass.run(*m_llvmModule);

    return { buffer.begin(), buffer.end() };
}


} // namespace sir