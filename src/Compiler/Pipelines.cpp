#include "Pipelines.hpp"

#include <Conversion/Passes.hpp>
#include <Dialect/BufferizationExtensions/Transforms/OneShotBufferizeCombined.hpp>
#include <Dialect/Stencil/Transforms/BufferizableOpInterfaceImpl.hpp>
#include <Dialect/Stencil/Transforms/Passes.hpp>
#include <Transforms/Passes.hpp>

#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/Arithmetic/Transforms/Passes.h>
#include <mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h>
#include <mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/Transforms/Passes.h>
#include <mlir/Transforms/Passes.h>


namespace sir {


Stage CreateBufferizationStage(mlir::MLIRContext& context) {
    Stage stage{ "bufferization", context };

    mlir::bufferization::OneShotBufferizationOptions bufferizationOptions;
    bufferizationOptions.allowUnknownOps = false;
    bufferizationOptions.allowReturnAllocs = false;
    bufferizationOptions.createDeallocs = true;
    bufferizationOptions.defaultMemorySpace = 0;
    bufferizationOptions.functionBoundaryTypeConversion = mlir::bufferization::BufferizationOptions::LayoutMapOption::FullyDynamicLayoutMap;
    bufferizationOptions.bufferizeFunctionBoundaries = true;

    stage.passes->addPass(createOneShotBufferizeCombinedPass(bufferizationOptions));
    stage.passes->addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createBufferizationBufferizePass());
    stage.passes->addNestedPass<mlir::func::FuncOp>(mlir::createTensorBufferizePass());
    stage.passes->addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createFinalizingBufferizePass());
    stage.passes->addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createBufferDeallocationPass());
    stage.passes->addPass(mlir::createCanonicalizerPass());
    stage.passes->addPass(mlir::createCSEPass());

    return stage;
}


Stage CreateGlobalOptimizationStage(mlir::MLIRContext& context,
                                    const OptimizationOptions& optimizationOptions) {
    Stage stage{ "global_opt", context };

    llvm::StringMap<mlir::OpPassManager> inlinerPipelines;
    const bool inlineFn = optimizationOptions.inlineFunctions;
    const bool fuseExtract = optimizationOptions.fuseExtractSliceOps;
    const bool fuseApply = optimizationOptions.fuseApplyOps;
    const bool elimAlloc = optimizationOptions.eliminateAllocBuffers;

    // Optimization pipeline for inliner
    mlir::OpPassManager inlinerFuncPm;
    inlinerFuncPm.addNestedPass<mlir::func::FuncOp>(createReduceDimOpsPass());
    inlinerFuncPm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
    inlinerFuncPm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
    elimAlloc ? inlinerFuncPm.addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createAllocTensorEliminationPass()) : void();
    inlinerPipelines.insert_or_assign("func.func", std::move(inlinerFuncPm));

    // General optimizer
    inlineFn ? stage.passes->addPass(mlir::createInlinerPass(std::move(inlinerPipelines))) : void();

    {
        auto& funcPrePm = stage.passes->nest<mlir::func::FuncOp>();

        funcPrePm.addPass(createReduceDimOpsPass());
        funcPrePm.addPass(mlir::createCSEPass());

        elimAlloc ? funcPrePm.addPass(mlir::bufferization::createAllocTensorEliminationPass()) : void();
        funcPrePm.addPass(createReduceDimOpsPass());
        funcPrePm.addPass(mlir::createCSEPass());
        funcPrePm.addPass(createEliminateSlicingPass());
    }

    fuseExtract ? stage.passes->addPass(createFuseExtractSliceOpsPass()) : void();
    fuseApply ? stage.passes->addPass(createFuseApplyOpsPass()) : void();

    {
        auto& funcPostPm = stage.passes->nest<mlir::func::FuncOp>();
        funcPostPm.addPass(createEliminateUnusedAllocTensorsPass());
        funcPostPm.addPass(mlir::createLoopInvariantCodeMotionPass());
        funcPostPm.addPass(mlir::createControlFlowSinkPass());
        funcPostPm.addPass(mlir::createCanonicalizerPass());
        funcPostPm.addPass(mlir::createCSEPass());
    }
    {
        auto& stencilPostPm = stage.passes->nest<stencil::StencilOp>();
        stencilPostPm.addPass(mlir::createLoopInvariantCodeMotionPass());
        stencilPostPm.addPass(mlir::createControlFlowSinkPass());
        stencilPostPm.addPass(mlir::createCanonicalizerPass());
        stencilPostPm.addPass(mlir::createCSEPass());
    }

    return stage;
}


std::vector<Stage> TargetCPUPipeline(mlir::MLIRContext& context,
                                     const OptimizationOptions& optimizationOptions) {
    Stage canonicalization{ "canonicalization", context };
    canonicalization.passes->addPass(mlir::createCSEPass());
    canonicalization.passes->addPass(mlir::createCanonicalizerPass());
    canonicalization.passes->addPass(createReduceDimOpsPass());
    canonicalization.passes->addPass(mlir::createTopologicalSortPass());

    Stage bufferization = CreateBufferizationStage(context);
    Stage globalOpt = CreateGlobalOptimizationStage(context, optimizationOptions);

    Stage loops{ "loops", context };
    loops.passes->addPass(createStencilApplyToLoopsPass());
    loops.passes->addPass(createStencilToFuncPass());

    Stage standard{ "standard", context };
    standard.passes->addPass(createStencilToStandardPass());
    standard.passes->addPass(createStencilPrintToLLVMPass());
    standard.passes->addPass(mlir::createCSEPass());
    standard.passes->addPass(mlir::createCanonicalizerPass());

    Stage llvm{ "llvm", context };
    llvm.passes->addPass(mlir::createConvertSCFToCFPass());
    llvm.passes->addPass(mlir::createConvertVectorToLLVMPass());
    llvm.passes->addPass(mlir::createMemRefToLLVMPass());
    llvm.passes->addPass(mlir::createConvertFuncToLLVMPass());
    llvm.passes->addPass(mlir::arith::createArithmeticExpandOpsPass());
    llvm.passes->addPass(mlir::arith::createConvertArithmeticToLLVMPass());
    llvm.passes->addPass(mlir::createConvertMathToLLVMPass());
    llvm.passes->addPass(mlir::createConvertMathToLibmPass());
    llvm.passes->addPass(mlir::cf::createConvertControlFlowToLLVMPass());
    llvm.passes->addPass(mlir::createCSEPass());
    llvm.passes->addPass(mlir::createCanonicalizerPass());

    std::array stages{ std::move(canonicalization),
                       std::move(globalOpt),
                       std::move(bufferization),
                       std::move(loops),
                       std::move(standard),
                       std::move(llvm) };
    return { std::move_iterator(stages.begin()), std::move_iterator(stages.end()) };
}


} // namespace sir