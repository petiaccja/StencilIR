#include "Pipelines.hpp"

#include <Conversion/Passes.hpp>
#include <Dialect/Stencil/Transforms/BufferizableOpInterfaceImpl.hpp>
#include <Dialect/Stencil/Transforms/Passes.hpp>
#include <Transforms/Passes.hpp>

#include <llvm/Support/TargetSelect.h>
#include <mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/Affine/Passes.h>
#include <mlir/Dialect/Arith/Transforms/Passes.h>
#include <mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h>
#include <mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h>
#include <mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/Passes.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/Dialect/MemRef/Transforms/Passes.h>
#include <mlir/Dialect/SCF/Transforms/Passes.h>
#include <mlir/Dialect/Tensor/Transforms/Passes.h>
#include <mlir/Transforms/Passes.h>


namespace sir {


Stage CreateCleanupStage(mlir::MLIRContext& context) {
    Stage stage{ "canonicalization", context };
    stage.passes->addPass(mlir::createCSEPass());
    stage.passes->addPass(mlir::createCanonicalizerPass());
    stage.passes->addPass(mlir::createTopologicalSortPass());
    return stage;
}


Stage CreateBufferizationStage(mlir::MLIRContext& context, int defaultMemorySpace) {
    Stage stage{ "bufferization", context };

    mlir::bufferization::OneShotBufferizationOptions bufferizationOptions;
    bufferizationOptions.allowUnknownOps = false;
    bufferizationOptions.allowReturnAllocs = false;
    bufferizationOptions.createDeallocs = true;
    bufferizationOptions.defaultMemorySpace = mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 64), defaultMemorySpace);
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
    inlinerPipelines.insert_or_assign(mlir::func::FuncOp::getOperationName(), std::move(inlinerFuncPm));

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


Stage CreateStencilToStandardStage(mlir::MLIRContext& context, bool runtimeVerification) {
    Stage stage{ "standard", context };
    runtimeVerification ? stage.passes->addPass(mlir::createGenerateRuntimeVerificationPass()) : void();
    stage.passes->addPass(createStencilToLoopsPass());
    stage.passes->addPass(createStencilToStandardPass());
    stage.passes->addPass(createStencilPrintToLLVMPass());
    stage.passes->addPass(mlir::createCSEPass());
    stage.passes->addPass(mlir::createCanonicalizerPass());
    return stage;
}


Stage CreateStandardToLLVMStage(mlir::MLIRContext& context, bool verify = true) {
    Stage stage{ "llvm", context };
    stage.passes->addPass(mlir::createConvertSCFToCFPass());

    stage.passes->addPass(mlir::memref::createExpandStridedMetadataPass());
    stage.passes->addPass(mlir::memref::createExpandOpsPass());
    stage.passes->addPass(mlir::createMemRefToLLVMConversionPass());

    stage.passes->addPass(mlir::createConvertVectorToLLVMPass());

    stage.passes->addPass(mlir::createLowerAffinePass());

    stage.passes->addPass(mlir::arith::createArithExpandOpsPass());
    stage.passes->addPass(mlir::createConvertMathToLibmPass());
    stage.passes->addPass(mlir::createConvertMathToLLVMPass());
    stage.passes->addPass(mlir::createArithToLLVMConversionPass());

    stage.passes->addPass(mlir::cf::createConvertControlFlowToLLVMPass());
    stage.passes->addPass(mlir::createConvertFuncToLLVMPass());

    stage.passes->addPass(mlir::createCanonicalizerPass());
    stage.passes->addPass(mlir::createCSEPass());
    verify ? stage.passes->addPass(mlir::createReconcileUnrealizedCastsPass()) : void();
    return stage;
}


std::vector<Stage> TargetCPUPipeline(mlir::MLIRContext& context,
                                     const OptimizationOptions& optimizationOptions) {
    Stage canonicalization = CreateCleanupStage(context);
    Stage globalOpt = CreateGlobalOptimizationStage(context, optimizationOptions);
    Stage func{ "func", context };
    func.passes->addPass(createStencilToFuncPass());
    Stage bufferization = CreateBufferizationStage(context, 0);
    Stage standard = CreateStencilToStandardStage(context, optimizationOptions.enableRuntimeVerification);
    Stage llvm = CreateStandardToLLVMStage(context);

    std::array stages{ std::move(canonicalization),
                       std::move(globalOpt),
                       std::move(func),
                       std::move(bufferization),
                       std::move(standard),
                       std::move(llvm) };
    return { std::make_move_iterator(stages.begin()), std::make_move_iterator(stages.end()) };
}


Stage CreateStandardToGPUStage(mlir::MLIRContext& context) {
    Stage stage{ "gpu", context };
    stage.passes->addNestedPass<mlir::func::FuncOp>(mlir::createParallelLoopTilingPass({ 256 }, true));
    stage.passes->addNestedPass<mlir::func::FuncOp>(mlir::createGpuMapParallelLoopsPass());
    stage.passes->addNestedPass<mlir::func::FuncOp>(mlir::createParallelLoopToGpuPass());
    stage.passes->addPass(mlir::createGpuKernelOutliningPass());
    stage.passes->addPass(mlir::createSymbolDCEPass());
    return stage;
}


Stage CreateCUDASerializerStage(mlir::MLIRContext& context, bool verify = true) {
#ifdef STENCILIR_ENABLE_CUDA
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();

    Stage stage{ "cuda_bin", context };
    stage.passes->addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createLowerGpuOpsToNVVMOpsPass());
    stage.passes->addPass(mlir::createCanonicalizerPass());
    stage.passes->addNestedPass<mlir::gpu::GPUModuleOp>(createUseCudaLibdevicePass());
    stage.passes->addNestedPass<mlir::gpu::GPUModuleOp>(mlir::createGpuSerializeToCubinPass("nvptx64-nvidia-cuda", "sm_35", "+ptx60"));
    stage.passes->addPass(mlir::createGpuToLLVMConversionPass());

    stage.passes->addPass(mlir::createCanonicalizerPass());
    stage.passes->addPass(mlir::createCSEPass());
    verify ? stage.passes->addPass(mlir::createReconcileUnrealizedCastsPass()) : void();

    return stage;
#else
    throw std::runtime_error("to use CUDA, Stencil IR must be compiled with STENCILIR_ENABLE_CUDA=ON");
#endif
}


std::vector<Stage> TargetCUDAPipeline(mlir::MLIRContext& context,
                                      const OptimizationOptions& optimizationOptions) {
    Stage canonicalization = CreateCleanupStage(context);
    Stage globalOpt = CreateGlobalOptimizationStage(context, optimizationOptions);
    Stage func{ "func", context };
    func.passes->addPass(createStencilToFuncPass());
    Stage bufferization = CreateBufferizationStage(context, 1);
    Stage standard = CreateStencilToStandardStage(context, false);
    Stage cuda = CreateStandardToGPUStage(context);
    Stage llvm = CreateStandardToLLVMStage(context, false);
    Stage serialize = CreateCUDASerializerStage(context, true);

    std::array stages{ std::move(canonicalization),
                       std::move(globalOpt),
                       std::move(func),
                       std::move(bufferization),
                       std::move(standard),
                       std::move(cuda),
                       std::move(llvm),
                       std::move(serialize) };
    return { std::make_move_iterator(stages.begin()), std::make_move_iterator(stages.end()) };
}


} // namespace sir