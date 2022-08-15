#include "StencilLoweringPasses.hpp"

#include <StencilDialect/StencilDialect.hpp>
#include <StencilDialect/StencilOps.hpp>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include <llvm/ADT/APFloat.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <memory>
#include <vector>


using namespace mlir;

struct KernelLowering : public OpRewritePattern<stencil::KernelOp> {
    using OpRewritePattern<stencil::KernelOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(stencil::KernelOp op, PatternRewriter& rewriter) const override final {
        Location loc = op->getLoc();

        const int64_t numDims = op.getNumDimensions().getSExtValue();

        std::vector<mlir::Type> functionParamTypes;
        const auto indexType = MemRefType::get({ numDims }, rewriter.getIndexType());
        functionParamTypes.push_back(indexType);
        for (auto& argument : op.getArgumentTypes()) {
            functionParamTypes.push_back(argument);
        }
        auto functionReturnTypes = op.getResultTypes();
        auto functionType = rewriter.getFunctionType(functionParamTypes, functionReturnTypes);

        auto funcOp = rewriter.create<func::FuncOp>(loc, op.getSymName(), functionType);
        rewriter.inlineRegionBefore(op.getRegion(), funcOp.getBody(), funcOp.end());
        Block& block = funcOp.getBody().front();

        // insertArgument seems buggy with empty list.
        block.getNumArguments() == 0 ? block.addArgument(indexType, loc)
                                     : block.insertArgument(block.args_begin(), indexType, loc);

        rewriter.eraseOp(op);
        return success();
    }
};

struct KernelReturnLowering : public OpRewritePattern<stencil::ReturnOp> {
    using OpRewritePattern<stencil::ReturnOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(stencil::ReturnOp op, PatternRewriter& rewriter) const override {
        Location loc = op->getLoc();
        rewriter.create<func::ReturnOp>(loc, op->getOperands());
        rewriter.eraseOp(op);
        return success();
    }
};

struct LaunchKernelLoweringBase : public OpRewritePattern<stencil::LaunchKernelOp> {
    using OpRewritePattern<stencil::LaunchKernelOp>::OpRewritePattern;

    static auto CreateLoopBody(stencil::LaunchKernelOp op, OpBuilder& builder, Location loc, ValueRange loopVars) {
        // Insert a kernel invocation within the loop
        std::vector<Type> resultTypes;
        for (const auto& target : op.getTargets()) {
            auto type = target.getType();
            assert(type.isa<MemRefType>());
            resultTypes.push_back(type.dyn_cast<MemRefType>().getElementType());
        }

        auto call = builder.create<stencil::InvokeKernelOp>(loc, resultTypes, op.getCallee(), loopVars, op.getArguments());

        // Store return values to targets
        const auto results = call.getResults();
        const auto targets = op.getTargets();
        const auto numResults = results.size();

        assert(results.size() == targets.size());

        for (size_t resultIdx = 0; resultIdx < numResults; ++resultIdx) {
            builder.create<AffineStoreOp>(loc, results[resultIdx], targets[resultIdx], loopVars);
        }
    };

    static auto GetGridSize(stencil::LaunchKernelOp op, OpBuilder& builder, Location loc)
        -> std::vector<Value> {
        const int64_t numDims = op.getGridDim().size();
        std::vector<Value> ubValues;
        for (ptrdiff_t boundIdx = 0; boundIdx < numDims; ++boundIdx) {
            ubValues.push_back(op.getGridDim()[boundIdx]);
        }
        return ubValues;
    }
};

struct LaunchKernelLoweringAffineLoop : LaunchKernelLoweringBase {
    using LaunchKernelLoweringBase::LaunchKernelLoweringBase;

    LogicalResult matchAndRewrite(stencil::LaunchKernelOp op, PatternRewriter& rewriter) const override final {
        Location loc = op->getLoc();
        const int64_t numDims = op.getGridDim().size();

        const auto ubValues = GetGridSize(op, rewriter, loc);
        const auto lbValues = std::vector<Value>(numDims, rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0)));
        const std::vector<int64_t> steps(numDims, 1);

        buildAffineLoopNest(rewriter, loc, lbValues, ubValues, steps, [&op](auto... args) { CreateLoopBody(op, args...); });
        rewriter.eraseOp(op);

        return success();
    }
};

struct LaunchKernelLoweringAffineParallel : LaunchKernelLoweringBase {
    using LaunchKernelLoweringBase::LaunchKernelLoweringBase;

    LogicalResult matchAndRewrite(stencil::LaunchKernelOp op, PatternRewriter& rewriter) const override final {
        Location loc = op->getLoc();
        const int64_t numDims = op.getGridDim().size();

        const auto ubValues = GetGridSize(op, rewriter, loc);
        const auto lbValues = std::vector<Value>(numDims, rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0)));
        const std::vector<int64_t> steps(numDims, 1);

        std::vector<AffineExpr> affineDims = {};
        for (ptrdiff_t i = 0; i < numDims; ++i) {
            affineDims.push_back(rewriter.getAffineDimExpr(i));
        }
        std::vector<AffineMap> affineMaps = {};
        for (ptrdiff_t i = 0; i < numDims; ++i) {
            affineMaps.push_back(AffineMap::get(numDims, 0, affineDims[i]));
        }
        std::array<Type, 0> resultTypes{};
        std::vector<arith::AtomicRMWKind> reductions = {};

        auto parallelOp = rewriter.create<AffineParallelOp>(loc, resultTypes, reductions, affineMaps, lbValues, affineMaps, ubValues, steps);
        auto& parallelBlock = *parallelOp.getBody();
        auto loopVars = parallelBlock.getArguments();
        rewriter.setInsertionPointToStart(&parallelBlock);
        CreateLoopBody(op, rewriter, loc, loopVars);

        rewriter.eraseOp(op);

        return success();
    }
};

struct LaunchKernelLoweringGPULaunch : LaunchKernelLoweringBase {
    using LaunchKernelLoweringBase::LaunchKernelLoweringBase;

    LogicalResult matchAndRewrite(stencil::LaunchKernelOp op, PatternRewriter& rewriter) const override final {
        Location loc = op->getLoc();
        const int64_t numDims = op.getGridDim().size();

        const auto domainSizes = GetGridSize(op, rewriter, loc);

        const auto blockSizes = [numDims]() -> std::array<int64_t, 3> {
            switch (numDims) {
                case 1: return { 256, 1, 1 };
                case 2: return { 16, 16, 1 };
                case 3: return { 8, 8, 4 };
                default: return { 8, 8, 4 };
            }
        }();

        Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        auto DivSnapUpwards = [&rewriter, &loc, &one](Value lhs, Value rhs) {
            Value decr = rewriter.create<arith::SubIOp>(loc, rhs, one);
            Value incr = rewriter.create<arith::AddIOp>(loc, lhs, decr);
            return Value(rewriter.create<arith::DivUIOp>(loc, incr, rhs));
        };

        auto totalSize = domainSizes;
        totalSize.resize(3, one);

        auto blockSizeX = rewriter.create<arith::ConstantIndexOp>(loc, blockSizes[0]);
        auto blockSizeY = rewriter.create<arith::ConstantIndexOp>(loc, blockSizes[1]);
        auto blockSizeZ = rewriter.create<arith::ConstantIndexOp>(loc, blockSizes[2]);

        auto gridSizeX = DivSnapUpwards(totalSize[0], blockSizeX);
        auto gridSizeY = DivSnapUpwards(totalSize[1], blockSizeY);
        auto gridSizeZ = DivSnapUpwards(totalSize[2], blockSizeZ);

        Value dynamicSharedMemorySize = nullptr;

        auto launchOp = rewriter.create<gpu::LaunchOp>(loc,
                                                       gridSizeX,
                                                       gridSizeY,
                                                       gridSizeZ,
                                                       blockSizeX,
                                                       blockSizeY,
                                                       blockSizeZ,
                                                       dynamicSharedMemorySize);

        Region& body = launchOp.body();
        Block& block = body.getBlocks().front();

        rewriter.setInsertionPointToEnd(&block);
        // TODO:
        // Pass upper bounds to block
        // Calculate absolute indices from threadIdx.? and blockIdx.?
        // Add affine::IfOp to halt threads outside bounds
        // Add affine::ForOp loops if dimension is larger than three
        rewriter.create<arith::ConstantIndexOp>(loc, 0);
        rewriter.create<gpu::TerminatorOp>(loc);

        rewriter.eraseOp(op);

        return success();
    }
};


struct InvokeKernelLowering : public OpRewritePattern<stencil::InvokeKernelOp> {
    using OpRewritePattern<stencil::InvokeKernelOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(stencil::InvokeKernelOp op, PatternRewriter& rewriter) const override {
        Location loc = op->getLoc();

        auto MakeIndex = [&rewriter, &loc](int64_t value) {
            return rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(value));
        };

        const int64_t numDims = op.getIndices().size();
        auto index = rewriter.create<memref::AllocaOp>(loc, MemRefType::get({ numDims }, rewriter.getIndexType()));
        for (ptrdiff_t dimIdx = 0; dimIdx < numDims; ++dimIdx) {
            rewriter.create<memref::StoreOp>(loc, op.getIndices()[dimIdx], index, ValueRange{ MakeIndex(dimIdx) });
        }

        std::vector<Value> operands;
        operands.push_back(index);
        for (const auto& argument : op.getArguments()) {
            operands.push_back(argument);
        }
        const auto resultTypes = op->getResultTypes();

        rewriter.replaceOpWithNewOp<func::CallOp>(op, op.getCallee(), resultTypes, operands);

        return success();
    }
};


struct IndexLowering : public OpRewritePattern<stencil::IndexOp> {
    using OpRewritePattern<stencil::IndexOp>::OpRewritePattern;

    LogicalResult match(stencil::IndexOp op) const override {
        auto parent = op->getParentOfType<func::FuncOp>();
        if (parent) {
            return success();
        }
        return failure();
    }

    void rewrite(stencil::IndexOp op, PatternRewriter& rewriter) const override {
        auto parent = op->getParentOfType<func::FuncOp>();
        assert(parent);

        const auto& blockArgs = parent.getBody().front().getArguments();
        const mlir::Value index = blockArgs.front();
        rewriter.replaceOp(op, { index });
    }
};

struct JumpLowering : public OpRewritePattern<stencil::JumpOp> {
    using OpRewritePattern<stencil::JumpOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(stencil::JumpOp op, PatternRewriter& rewriter) const override final {
        Location loc = op->getLoc();

        Value index = op.getInputIndex();
        std::vector<int64_t> offsets;
        for (const auto& offset : op.getOffset().getAsRange<IntegerAttr>()) {
            offsets.push_back(offset.getInt());
        }

        assert(index.getType().isa<MemRefType>());
        const auto indexType = index.getType().dyn_cast<MemRefType>();

        Value result = rewriter.create<memref::AllocaOp>(loc, indexType);
        for (size_t dimIdx = 0; dimIdx < offsets.size(); ++dimIdx) {
            std::array accessIndex = { Value(rewriter.create<arith::ConstantIndexOp>(loc, dimIdx)) };
            Value original = rewriter.create<memref::LoadOp>(loc, index, accessIndex);
            Value offset = rewriter.create<arith::ConstantIndexOp>(loc, offsets[dimIdx]);
            Value sum = rewriter.create<arith::AddIOp>(loc, original, offset);
            rewriter.create<memref::StoreOp>(loc, sum, result, accessIndex);
        }

        rewriter.replaceOp(op, result);

        return success();
    }
};

struct SampleLowering : public OpRewritePattern<stencil::SampleOp> {
    using OpRewritePattern<stencil::SampleOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(stencil::SampleOp op, PatternRewriter& rewriter) const override final {
        Location loc = op->getLoc();

        Value index = op.getIndex();
        Value field = op.getField();

        assert(index.getType().isa<MemRefType>());
        const auto indexType = index.getType().dyn_cast<MemRefType>();
        const auto numDims = indexType.getShape()[0];
        assert(numDims != ShapedType::kDynamicSize);

        std::vector<Value> indices;
        for (ptrdiff_t dimIdx = 0; dimIdx < numDims; ++dimIdx) {
            std::array loadIndex = { Value(rewriter.create<arith::ConstantIndexOp>(loc, dimIdx)) };
            indices.push_back(rewriter.create<memref::LoadOp>(loc, index, loadIndex));
        }

        Value value = rewriter.create<memref::LoadOp>(loc, field, indices);
        rewriter.replaceOp(op, value);

        return success();
    }
};


struct JumpIndirectLowering : public OpRewritePattern<stencil::JumpIndirectOp> {
    using OpRewritePattern<stencil::JumpIndirectOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(stencil::JumpIndirectOp op, PatternRewriter& rewriter) const override final {
        Location loc = op->getLoc();

        auto inputIndex = op.getInputIndex();
        const auto dimension = op.getDimension().getSExtValue();
        std::array<Value, 1> dimIndices = { rewriter.create<arith::ConstantIndexOp>(loc, dimension) };
        Value inputIndexElem = rewriter.create<memref::LoadOp>(loc, inputIndex, dimIndices);

        auto map = op.getMap();
        auto mapElement = op.getMapElement();
        std::array<Value, 2> mapIndices = { inputIndexElem, mapElement };
        Value newIndexElem = rewriter.create<memref::LoadOp>(loc, map, mapIndices);

        assert(inputIndex.getType().isa<MemRefType>());
        const auto indexType = inputIndex.getType().dyn_cast<MemRefType>();

        Value outputIndex = rewriter.create<memref::AllocaOp>(loc, indexType);
        rewriter.create<memref::CopyOp>(loc, inputIndex, outputIndex);
        rewriter.create<memref::StoreOp>(loc, newIndexElem, outputIndex, dimIndices);

        rewriter.replaceOp(op, outputIndex);

        return success();
    }
};


struct SampleIndirectLowering : public OpRewritePattern<stencil::SampleIndirectOp> {
    using OpRewritePattern<stencil::SampleIndirectOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(stencil::SampleIndirectOp op, PatternRewriter& rewriter) const override final {
        Location loc = op->getLoc();

        auto inputIndex = op.getIndex();
        const auto dimension = op.getDimension().getSExtValue();
        std::array<Value, 1> dimIndices = { rewriter.create<arith::ConstantIndexOp>(loc, dimension) };
        Value inputIndexElem = rewriter.create<memref::LoadOp>(loc, inputIndex, dimIndices);

        auto field = op.getField();
        auto fieldElement = op.getFieldElement();
        std::array<Value, 2> mapIndices = { inputIndexElem, fieldElement };
        Value sample = rewriter.create<memref::LoadOp>(loc, field, mapIndices);

        rewriter.replaceOp(op, sample);

        return success();
    }
};


void StencilToAffinePass::getDependentDialects(DialectRegistry& registry) const {
    registry.insert<arith::ArithmeticDialect, AffineDialect, memref::MemRefDialect>();
}


void StencilToAffinePass::runOnOperation() {
    ConversionTarget target(getContext());
    target.addLegalDialect<AffineDialect>();
    target.addLegalDialect<arith::ArithmeticDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<stencil::StencilDialect>();
    target.addIllegalOp<stencil::LaunchKernelOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<LaunchKernelLoweringAffineLoop>(&getContext(), 1, ArrayRef<StringRef>{});

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}


void StencilToFuncPass::getDependentDialects(DialectRegistry& registry) const {
    registry.insert<arith::ArithmeticDialect, AffineDialect, memref::MemRefDialect>();
}


void StencilToFuncPass::runOnOperation() {
    ConversionTarget target(getContext());
    target.addLegalDialect<AffineDialect>();
    target.addLegalDialect<arith::ArithmeticDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addIllegalDialect<stencil::StencilDialect>();
    target.addLegalOp<stencil::PrintOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<KernelLowering>(&getContext());
    patterns.add<KernelReturnLowering>(&getContext());
    patterns.add<InvokeKernelLowering>(&getContext());

    patterns.add<IndexLowering>(&getContext());
    patterns.add<JumpLowering>(&getContext());
    patterns.add<SampleLowering>(&getContext());
    patterns.add<JumpIndirectLowering>(&getContext());
    patterns.add<SampleIndirectLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}


void StencilToGPUPass::getDependentDialects(DialectRegistry& registry) const {
    registry.insert<arith::ArithmeticDialect,
                    AffineDialect,
                    memref::MemRefDialect,
                    stencil::StencilDialect,
                    gpu::GPUDialect>();
}


void StencilToGPUPass::runOnOperation() {
    ConversionTarget target(getContext());
    target.addLegalDialect<AffineDialect>();
    target.addLegalDialect<gpu::GPUDialect>();
    target.addLegalDialect<arith::ArithmeticDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<stencil::StencilDialect>();
    target.addIllegalOp<stencil::LaunchKernelOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<LaunchKernelLoweringGPULaunch>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}