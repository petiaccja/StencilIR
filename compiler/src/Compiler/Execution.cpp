#include "Execution.hpp"

#include <llvm/Support/TargetSelect.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>


JitRunner::JitRunner(mlir::ModuleOp& llvmIr) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::registerLLVMDialectTranslation(*llvmIr.getContext());

    constexpr int optLevel = 0;
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
}