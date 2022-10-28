#include "Execution.hpp"

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

    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;
    auto maybeEngine = mlir::ExecutionEngine::create(llvmIr, engineOptions);
    if (!maybeEngine) {
        throw std::runtime_error("failed to construct an execution engine");
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
    // TODO: this shit's gonna be failing until the MLIR execution engine
    //  and/or all support stuff is --whole-archive linked into the DLL or whatnot.
    llvm::Error error = m_engine->invokePacked(name, { args.data(), args.size() });
    if (error) {
        std::string message;
        error = llvm::handleErrors(std::move(error), [&](llvm::ErrorInfoBase& err) {
            message = err.message();
        });
        if (error) {
            throw std::runtime_error("Error while handling errors. Seriously, what the fuck is this?");
        }
        throw std::runtime_error("Invoking JIT-ed function failed: " + message);
    }
}