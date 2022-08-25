#include "Pipelines.hpp"

#include <Conversion/Passes.hpp>
#include <Dialect/Stencil/Transforms/BufferizableOpInterfaceImpl.hpp>

#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h>
#include <mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/Transforms/Passes.h>
#include <mlir/Transforms/Passes.h>


Stage CreateBufferizationStage(mlir::MLIRContext& context) {
    Stage stage{ "bufferization", context };

    mlir::DialectRegistry registry;
    mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
    stencil::registerBufferizableOpInterfaceExternalModels(registry);
    context.appendDialectRegistry(registry);

    mlir::bufferization::OneShotBufferizationOptions bufferizationOptions;
    bufferizationOptions.allowUnknownOps = false;
    bufferizationOptions.allowReturnAllocs = false;
    bufferizationOptions.createDeallocs = false;
    bufferizationOptions.defaultMemorySpace = 0;
    bufferizationOptions.functionBoundaryTypeConversion = mlir::bufferization::BufferizationOptions::LayoutMapOption::FullyDynamicLayoutMap;

    mlir::bufferization::OneShotBufferizationOptions funcBufferizationOptions = bufferizationOptions;
    funcBufferizationOptions.bufferizeFunctionBoundaries = true;
    funcBufferizationOptions.opFilter.allowOperation([](mlir::Operation* op) -> bool {
        return mlir::isa<mlir::func::FuncOp>(op) || op->getParentOfType<mlir::func::FuncOp>();
    });

    mlir::bufferization::OneShotBufferizationOptions indepBufferizationOptions = bufferizationOptions;
    indepBufferizationOptions.opFilter.denyOperation<mlir::func::FuncOp>();

    stage.passes->addPass(mlir::bufferization::createOneShotBufferizePass(funcBufferizationOptions));
    stage.passes->addPass(mlir::bufferization::createOneShotBufferizePass(indepBufferizationOptions));
    stage.passes->addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createBufferizationBufferizePass());
    stage.passes->addNestedPass<mlir::func::FuncOp>(mlir::createTensorBufferizePass());
    stage.passes->addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createFinalizingBufferizePass());
    stage.passes->addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createBufferHoistingPass());
    stage.passes->addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createBufferLoopHoistingPass());
    stage.passes->addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createBufferDeallocationPass());

    // Verifier must be disabled for this pass because it's rather hacky with the separated one-shot bufferize.
    stage.passes->enableVerifier(false);

    return stage;
}


std::vector<Stage> TargetCPUPipeline(mlir::MLIRContext& context) {
    Stage canonicalization{ "canonicalization", context };
    canonicalization.passes->addPass(mlir::createCanonicalizerPass());
    canonicalization.passes->addPass(mlir::createCSEPass());
    canonicalization.passes->addPass(mlir::createTopologicalSortPass());

    Stage bufferization = CreateBufferizationStage(context);

    Stage loops{ "loops", context };
    loops.passes->addPass(createStencilApplyToLoopsPass());

    Stage standard{ "standard", context };
    standard.passes->addPass(createStencilOpsToStandardPass());
    standard.passes->addPass(createStencilPrintToLLVMPass());
    standard.passes->addPass(mlir::createCanonicalizerPass());
    standard.passes->addPass(mlir::createCSEPass());

    Stage llvm{ "llvm", context };
    llvm.passes->addPass(mlir::createConvertSCFToCFPass());
    llvm.passes->addPass(mlir::createConvertVectorToLLVMPass());
    llvm.passes->addPass(mlir::createMemRefToLLVMPass());
    llvm.passes->addPass(mlir::createConvertFuncToLLVMPass());
    llvm.passes->addPass(mlir::arith::createConvertArithmeticToLLVMPass());
    llvm.passes->addPass(mlir::cf::createConvertControlFlowToLLVMPass());
    llvm.passes->addPass(mlir::createCanonicalizerPass());
    llvm.passes->addPass(mlir::createCSEPass());

    std::array stages{ std::move(canonicalization),
                       std::move(bufferization),
                       std::move(loops),
                       std::move(standard),
                       std::move(llvm) };
    return { std::move_iterator(stages.begin()), std::move_iterator(stages.end()) };
}