#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

namespace mlir {

namespace bufferization {
    class BufferizationDialect;
} // namespace bufferization

namespace func {
    class FuncDialect;
} // namespace func

namespace memref {
    class MemRefDialect;
} // namespace memref

#define GEN_PASS_CLASSES
#include "mlir/Dialect/Bufferization/Transforms/Passes.h.inc"

} // namespace mlir


#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Bufferization/Transforms/Bufferize.h>
#include <mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h>
#include <mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Transforms/Passes.h>


using namespace mlir;
using namespace bufferization;


static LayoutMapOption parseLayoutMapOption(const std::string& s) {
    if (s == "fully-dynamic-layout-map")
        return LayoutMapOption::FullyDynamicLayoutMap;
    if (s == "identity-layout-map")
        return LayoutMapOption::IdentityLayoutMap;
    if (s == "infer-layout-map")
        return LayoutMapOption::InferLayoutMap;
    llvm_unreachable("invalid layout map option");
}


struct OneShotBufferizeCombinedPass
    : public OneShotBufferizeBase<OneShotBufferizeCombinedPass> {
    OneShotBufferizeCombinedPass() : OneShotBufferizeBase<OneShotBufferizeCombinedPass>() {}

    explicit OneShotBufferizeCombinedPass(const OneShotBufferizationOptions& options)
        : options(options) {}

    void getDependentDialects(DialectRegistry& registry) const override {
        registry
            .insert<bufferization::BufferizationDialect, memref::MemRefDialect>();
        registerAllocationOpInterfaceExternalModels(registry);
    }

    void runOnOperation() override {
        OneShotBufferizationOptions opt;
        if (!options) {
            // Make new bufferization options if none were provided when creating the
            // pass.
            opt.allowReturnAllocs = allowReturnAllocs;
            opt.allowUnknownOps = allowUnknownOps;
            opt.analysisFuzzerSeed = analysisFuzzerSeed;
            opt.createDeallocs = createDeallocs;
            opt.functionBoundaryTypeConversion =
                parseLayoutMapOption(functionBoundaryTypeConversion);
            if (mustInferMemorySpace)
                opt.defaultMemorySpace = {};
            opt.printConflicts = printConflicts;
            opt.testAnalysisOnly = testAnalysisOnly;
            opt.bufferizeFunctionBoundaries = bufferizeFunctionBoundaries;

            // Configure type converter.
            LayoutMapOption unknownTypeConversionOption =
                parseLayoutMapOption(unknownTypeConversion);
            opt.unknownTypeConverterFn = [=](Value value, unsigned memorySpace,
                                             const BufferizationOptions& options) {
                auto tensorType = value.getType().cast<TensorType>();
                if (unknownTypeConversionOption == LayoutMapOption::IdentityLayoutMap)
                    return bufferization::getMemRefTypeWithStaticIdentityLayout(
                        tensorType, memorySpace);
                assert(
                    unknownTypeConversionOption == LayoutMapOption::FullyDynamicLayoutMap && "invalid layout map option");
                return bufferization::getMemRefTypeWithFullyDynamicLayout(tensorType,
                                                                          memorySpace);
            };

            // Configure op filter.
            OpFilter::Entry::FilterFn filterFn = [&](Operation* op) {
                // Filter may be specified via options.
                if (this->dialectFilter.hasValue())
                    return llvm::is_contained(this->dialectFilter,
                                              op->getDialect()->getNamespace());
                // No filter specified: All other ops are allowed.
                return true;
            };
            opt.opFilter.allowOperation(filterFn);
        }
        else {
            opt = *options;
        }

        ModuleOp moduleOp = getOperation();
        if (opt.bufferizeFunctionBoundaries) {
            // Bufferize only function operations inside moduleOp.
            auto funcOpt = opt;
            funcOpt.opFilter.allowOperation([](mlir::Operation* op) -> bool {
                return mlir::isa<mlir::func::FuncOp>(op) || op->getParentOfType<mlir::func::FuncOp>();
            });

            if (failed(runOneShotModuleBufferize(moduleOp, funcOpt))) {
                signalPassFailure();
                return;
            }

            // Bufferize everything else except functions.
            auto restOpt = opt;
            restOpt.bufferizeFunctionBoundaries = false;
            restOpt.opFilter.denyOperation([](mlir::Operation* op) -> bool {
                return mlir::isa<mlir::func::FuncOp>(op) || op->getParentOfType<mlir::func::FuncOp>();
            });

            if (failed(runOneShotBufferize(moduleOp, restOpt))) {
                signalPassFailure();
                return;
            }
        }
        else {
            if (failed(runOneShotBufferize(moduleOp, opt))) {
                signalPassFailure();
                return;
            }
        }

        if (opt.testAnalysisOnly)
            return;

        OpPassManager cleanupPipeline("builtin.module");
        cleanupPipeline.addPass(createCanonicalizerPass());
        cleanupPipeline.addPass(createCSEPass());
        cleanupPipeline.addPass(createLoopInvariantCodeMotionPass());
        (void)runPipeline(cleanupPipeline, moduleOp);
    }

private:
    llvm::Optional<OneShotBufferizationOptions> options;
};


std::unique_ptr<mlir::Pass> createOneShotBufferizeCombinedPass(const mlir::bufferization::OneShotBufferizationOptions& options) {
    return std::make_unique<OneShotBufferizeCombinedPass>(options);
}