#include "AllToLLVMPass.hpp"
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



mlir::ModuleOp TestProgram(mlir::MLIRContext& context) {
    context.getOrLoadDialect<mock::MockDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();

    mlir::OpBuilder builder{ &context };

    auto module = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());

    builder.setInsertionPointToEnd(module.getBody());

    auto mainFuncType = mlir::FunctionType::get(&context, llvm::ArrayRef<mlir::Type>{}, llvm::ArrayRef<mlir::Type>{});
    auto mainFunc = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(), "main", mainFuncType);
    auto& mainFuncBlock = mainFunc.getBody().emplaceBlock();

    builder.setInsertionPointToStart(&mainFuncBlock);
    const mlir::Value c1 = builder.create<mock::ConstantOp>(builder.getUnknownLoc(), builder.getF32Type(), builder.getFloatAttr(builder.getF32Type(), 1.5f));
    builder.create<mock::PrintOp>(builder.getUnknownLoc(), c1);
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());

    return module;
}

mlir::LogicalResult LowerProgram(mlir::MLIRContext& context, mlir::ModuleOp& op) {
    mlir::PassManager passManager(&context);
    passManager.addPass(std::make_unique<MockToArithPass>());
    passManager.addPass(std::make_unique<MockPrintPass>());
    passManager.addPass(std::make_unique<AllToLLVMPass>());
    return passManager.run(op);
}

bool ExecuteProgram(mlir::ModuleOp& module) {
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
    auto invocationResult = engine->invokePacked("main");
    if (invocationResult) {
        llvm::errs() << "JIT invocation failed\n";
        return false;
    }

    return true;
}

int main() {
    mlir::MLIRContext context;

    mlir::ModuleOp module = TestProgram(context);
    std::cout << "Mock IR:\n"
              << std::endl;
    module->dump();

    if (LowerProgram(context, module).succeeded()) {
        std::cout << "\n\nLLVM IR:\n"
                  << std::endl;
        module->dump();
        if (!ExecuteProgram(module)) {
            std::cout << "failed to run!" << std::endl;
        }
    }
    else {
        std::cout << "failed to lower!" << std::endl;
    }
}