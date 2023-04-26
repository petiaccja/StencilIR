#include "Pipelines.hpp"

#include <Conversion/Passes.hpp>
#include <Dialect/Stencil/Transforms/BufferizableOpInterfaceImpl.hpp>
#include <Dialect/Stencil/Transforms/Passes.hpp>
#include <Transforms/Passes.hpp>

#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/Affine/Passes.h>
#include <mlir/Dialect/Arith/Transforms/Passes.h>
#include <mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h>
#include <mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/Passes.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Dialect/Tensor/Transforms/Passes.h>
#include <mlir/Transforms/Passes.h>


namespace sir {


Stage CreateBufferizationStage(mlir::MLIRContext& context) {
    Stage stage{ "bufferization", context };

    mlir::bufferization::OneShotBufferizationOptions bufferizationOptions;
    bufferizationOptions.allowUnknownOps = false;
    bufferizationOptions.allowReturnAllocs = false;
    bufferizationOptions.createDeallocs = true;
    bufferizationOptions.defaultMemorySpace = mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 64), 0);
    bufferizationOptions.functionBoundaryTypeConversion = mlir::bufferization::LayoutMapOption::FullyDynamicLayoutMap;
    bufferizationOptions.bufferizeFunctionBoundaries = true;

    stage.passes->addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createEmptyTensorToAllocTensorPass());
    stage.passes->addPass(mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
    stage.passes->addNestedPass<mlir::func::FuncOp>(mlir::createTensorBufferizePass());
    stage.passes->addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createBufferizationBufferizePass());
    stage.passes->addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createFinalizingBufferizePass());
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
    const bool elimSlicing = true; // Maybe expose / run always?
    const bool redDims = true; // Maybe expose / run always?

    // Optimization pipeline for inliner
    mlir::OpPassManager inlinerFuncPm;
    redDims ? inlinerFuncPm.addNestedPass<mlir::func::FuncOp>(createReduceDimOpsPass()) : void();
    inlinerFuncPm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
    inlinerFuncPm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
    elimAlloc ? inlinerFuncPm.addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createEmptyTensorEliminationPass()) : void();
    inlinerPipelines.insert_or_assign("func.func", std::move(inlinerFuncPm));

    // General optimizer
    // Temporarily here, where it causes no issues
    elimAlloc ? stage.passes->addNestedPass<mlir::func::FuncOp>(mlir::bufferization::createEmptyTensorEliminationPass()) : void();

    inlineFn ? stage.passes->addPass(mlir::createInlinerPass(std::move(inlinerPipelines))) : void();

    {
        auto& funcPrePm = stage.passes->nest<mlir::func::FuncOp>();

        redDims ? funcPrePm.addPass(createReduceDimOpsPass()) : void();
        funcPrePm.addPass(mlir::createCanonicalizerPass());
        funcPrePm.addPass(mlir::createCSEPass());

        // Better here, but the pass has a bug
        // elimAlloc ? funcPrePm.addPass(mlir::bufferization::createEmptyTensorEliminationPass()) : void();
        redDims ? funcPrePm.addPass(createReduceDimOpsPass()) : void();
        funcPrePm.addPass(mlir::createCanonicalizerPass());
        funcPrePm.addPass(mlir::createCSEPass());
        elimSlicing ? funcPrePm.addPass(createEliminateSlicingPass()) : void();
    }

    fuseExtract ? stage.passes->addPass(createFuseExtractSliceOpsPass()) : void();
    fuseApply ? stage.passes->addPass(createFuseApplyOpsPass()) : void();

    {
        auto& funcPostPm = stage.passes->nest<mlir::func::FuncOp>();
        funcPostPm.addPass(mlir::createCanonicalizerPass());
        funcPostPm.addPass(mlir::createCSEPass());
        funcPostPm.addPass(createEliminateUnusedAllocTensorsPass());
        funcPostPm.addPass(mlir::createLoopInvariantCodeMotionPass());
        funcPostPm.addPass(mlir::createControlFlowSinkPass());
    }
    {
        auto& stencilPostPm = stage.passes->nest<stencil::StencilOp>();
        stencilPostPm.addPass(mlir::createCanonicalizerPass());
        stencilPostPm.addPass(mlir::createCSEPass());
        stencilPostPm.addPass(mlir::createLoopInvariantCodeMotionPass());
        stencilPostPm.addPass(mlir::createControlFlowSinkPass());
    }

    return stage;
}


std::vector<Stage> TargetCPUPipeline(mlir::MLIRContext& context,
                                     const OptimizationOptions& optimizationOptions) {
    // Clean-up
    Stage canonicalization{ "canonicalization", context };
    canonicalization.passes->addPass(mlir::createCSEPass());
    canonicalization.passes->addPass(mlir::createCanonicalizerPass());
    canonicalization.passes->addPass(mlir::createTopologicalSortPass());

    // High-level optimizer
    Stage globalOpt = CreateGlobalOptimizationStage(context, optimizationOptions);

    // Convert to func
    Stage func{ "func", context };
    func.passes->addPass(createStencilToFuncPass());

    // Bufferize
    Stage bufferization = CreateBufferizationStage(context);

    // Convert out of stencil dialect
    Stage standard{ "standard", context };
    standard.passes->addPass(createStencilToLoopsPass());
    standard.passes->addPass(createStencilToStandardPass());
    standard.passes->addPass(createStencilPrintToLLVMPass());
    standard.passes->addPass(mlir::createCSEPass());
    standard.passes->addPass(mlir::createCanonicalizerPass());

    // Convert all to LLVM IR
    Stage llvm{ "llvm", context };
    llvm.passes->addPass(mlir::createConvertSCFToCFPass());

    llvm.passes->addPass(mlir::memref::createExpandStridedMetadataPass());
    llvm.passes->addPass(mlir::memref::createExpandOpsPass());
    llvm.passes->addPass(mlir::createMemRefToLLVMConversionPass());

    llvm.passes->addPass(mlir::createConvertVectorToLLVMPass());

    llvm.passes->addPass(mlir::createLowerAffinePass());

    llvm.passes->addPass(mlir::arith::createArithExpandOpsPass());
    llvm.passes->addPass(mlir::createConvertMathToLibmPass());
    llvm.passes->addPass(mlir::createConvertMathToLLVMPass());
    llvm.passes->addPass(mlir::createArithToLLVMConversionPass());

    llvm.passes->addPass(mlir::cf::createConvertControlFlowToLLVMPass());
    llvm.passes->addPass(mlir::createConvertFuncToLLVMPass());

    llvm.passes->addPass(mlir::createCanonicalizerPass());
    llvm.passes->addPass(mlir::createCSEPass());
    llvm.passes->addPass(mlir::createReconcileUnrealizedCastsPass());

    std::array stages{ std::move(canonicalization),
                       std::move(globalOpt),
                       std::move(func),
                       std::move(bufferization),
                       std::move(standard),
                       std::move(llvm) };
    return { std::make_move_iterator(stages.begin()), std::make_move_iterator(stages.end()) };
}


} // namespace sir