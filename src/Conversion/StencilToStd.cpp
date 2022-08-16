#include "StencilToStd.hpp"

#include <StencilDialect/StencilDialect.hpp>
#include <StencilDialect/StencilOps.hpp>

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/ArrayRef.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LLVM.h>
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

struct StencilOpLowering : public OpRewritePattern<stencil::StencilOp> {
    using OpRewritePattern<stencil::StencilOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(stencil::StencilOp op, PatternRewriter& rewriter) const override final {
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

struct ReturnOpLowering : public OpRewritePattern<stencil::ReturnOp> {
    using OpRewritePattern<stencil::ReturnOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(stencil::ReturnOp op, PatternRewriter& rewriter) const override {
        Location loc = op->getLoc();
        rewriter.create<func::ReturnOp>(loc, op->getOperands());
        rewriter.eraseOp(op);
        return success();
    }
};

struct LaunchKernelLoweringBase : public OpRewritePattern<stencil::ApplyOp> {
    using OpRewritePattern<stencil::ApplyOp>::OpRewritePattern;

    static auto CreateLoopBody(stencil::ApplyOp op, OpBuilder& builder, Location loc, ValueRange loopVars) {
        // Insert a kernel invocation within the loop
        std::vector<Type> resultTypes;
        for (const auto& target : op.getOutputs()) {
            auto type = target.getType();
            assert(type.isa<MemRefType>());
            resultTypes.push_back(type.dyn_cast<MemRefType>().getElementType());
        }

        auto call = builder.create<stencil::InvokeStencilOp>(loc, resultTypes, op.getCallee(), loopVars, op.getInputs());

        // Store return values to targets
        const auto results = call.getResults();
        const auto outputs = op.getOutputs();
        const auto numResults = results.size();

        assert(results.size() == outputs.size());

        for (size_t resultIdx = 0; resultIdx < numResults; ++resultIdx) {
            builder.create<memref::StoreOp>(loc, results[resultIdx], outputs[resultIdx], loopVars);
        }
    };

    static int64_t GetDimension(stencil::ApplyOp op) {
        const auto& output0 = op.getOutputs()[0];
        const auto& type0 = output0.getType().cast<mlir::ShapedType>();
        return type0.getRank();
    }

    static auto GetOutputSize(stencil::ApplyOp op, OpBuilder& builder, Location loc)
        -> std::vector<Value> {
        const auto& output0 = op.getOutputs()[0];
        const int64_t numDims = GetDimension(op);
        std::vector<Value> ubValues;
        for (ptrdiff_t dimIdx = 0; dimIdx < numDims; ++dimIdx) {
            ubValues.push_back(builder.create<memref::DimOp>(loc, output0, dimIdx));
        }
        return ubValues;
    }
};

struct ApplyOpLoweringSCF : LaunchKernelLoweringBase {
    using LaunchKernelLoweringBase::LaunchKernelLoweringBase;

    LogicalResult matchAndRewrite(stencil::ApplyOp op, PatternRewriter& rewriter) const override final {
        Location loc = op->getLoc();
        const int64_t numDims = GetDimension(op);

        const auto ubValues = GetOutputSize(op, rewriter, loc);
        const auto lbValues = std::vector<Value>(numDims, rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0)));
        const std::vector<Value> steps(numDims, rewriter.create<arith::ConstantIndexOp>(loc, 1));

        scf::buildLoopNest(rewriter, loc, lbValues, ubValues, steps, [&op](OpBuilder& builder, Location loc, ValueRange loopVars) {
            CreateLoopBody(op, builder, loc, loopVars);
        });
        rewriter.replaceOp(op, op.getOutputs());

        return success();
    }
};


struct ApplyOpLoweringGPULaunch : LaunchKernelLoweringBase {
    using LaunchKernelLoweringBase::LaunchKernelLoweringBase;

    LogicalResult matchAndRewrite(stencil::ApplyOp op, PatternRewriter& rewriter) const override final {
        Location loc = op->getLoc();
        const int64_t numDims = GetDimension(op);

        const auto domainSizes = GetOutputSize(op, rewriter, loc);

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

        rewriter.replaceOp(op, op.getOutputs());

        return success();
    }
};


struct InvokeStencilLowering : public OpRewritePattern<stencil::InvokeStencilOp> {
    using OpRewritePattern<stencil::InvokeStencilOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(stencil::InvokeStencilOp op, PatternRewriter& rewriter) const override {
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


struct IndexOpLowering : public OpRewritePattern<stencil::IndexOp> {
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

struct JumpOpLowering : public OpRewritePattern<stencil::JumpOp> {
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

struct SampleOpLowering : public OpRewritePattern<stencil::SampleOp> {
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


struct JumpIndirectOpLowering : public OpRewritePattern<stencil::JumpIndirectOp> {
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


struct SampleIndirectOpLowering : public OpRewritePattern<stencil::SampleIndirectOp> {
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


void StencilToStdPass::getDependentDialects(DialectRegistry& registry) const {
    registry.insert<arith::ArithmeticDialect,
                    func::FuncDialect,
                    memref::MemRefDialect,
                    scf::SCFDialect>();
}


void StencilToStdPass::runOnOperation() {
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithmeticDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addIllegalDialect<stencil::StencilDialect>();
    target.addLegalOp<stencil::PrintOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<StencilOpLowering>(&getContext());
    patterns.add<ReturnOpLowering>(&getContext());
    patterns.add<InvokeStencilLowering>(&getContext());
    if (launchToGpu) {
        patterns.add<ApplyOpLoweringGPULaunch>(&getContext());
    }
    else {
        patterns.add<ApplyOpLoweringSCF>(&getContext());
    }

    patterns.add<IndexOpLowering>(&getContext());
    patterns.add<JumpOpLowering>(&getContext());
    patterns.add<SampleOpLowering>(&getContext());
    patterns.add<JumpIndirectOpLowering>(&getContext());
    patterns.add<SampleIndirectOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}