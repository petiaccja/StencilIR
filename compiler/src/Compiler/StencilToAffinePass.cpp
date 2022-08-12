#include "StencilToAffinePass.hpp"

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
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <algorithm>
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

struct KernelCallLowering : public OpRewritePattern<stencil::LaunchKernelOp> {
    bool m_makeParallelLoops = false;

    KernelCallLowering(MLIRContext* context,
                       PatternBenefit benefit = 1,
                       ArrayRef<StringRef> generatedNames = {},
                       bool makeParallelLoops = false)
        : OpRewritePattern<stencil::LaunchKernelOp>(context, benefit, generatedNames),
          m_makeParallelLoops(makeParallelLoops) {}

    LogicalResult matchAndRewrite(stencil::LaunchKernelOp op, PatternRewriter& rewriter) const override final {
        Location loc = op->getLoc();
        const int64_t numDims = op.getGridDim().size();

        auto createLoopBody = [&op, numDims](OpBuilder& builder, Location loc, ValueRange loopVars) {
            auto MakeIndex = [&builder, &loc](int64_t value) {
                return builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(value));
            };

            // Make operands list: { index, args... }
            auto index = builder.create<memref::AllocaOp>(loc, MemRefType::get({ numDims }, builder.getIndexType()));
            for (ptrdiff_t dimIdx = 0; dimIdx < numDims; ++dimIdx) {
                builder.create<memref::StoreOp>(loc, loopVars[dimIdx], index, ValueRange{ MakeIndex(dimIdx) });
            }

            std::vector<Value> operands;
            operands.push_back(index);
            for (const auto& argument : op.getArguments()) {
                operands.push_back(argument);
            }

            std::vector<Type> resultTypes;
            for (const auto& target : op.getTargets()) {
                auto type = target.getType();
                assert(type.isa<MemRefType>());
                resultTypes.push_back(type.dyn_cast<MemRefType>().getElementType());
            }

            // Create regular function call to kernel
            auto call = builder.create<func::CallOp>(loc, op.getCallee(), resultTypes, operands);

            // Store return values to targets
            const auto results = call.getResults();
            const auto targets = op.getTargets();
            const auto numResults = results.size();

            assert(results.size() == targets.size());

            for (size_t resultIdx = 0; resultIdx < numResults; ++resultIdx) {
                builder.create<memref::StoreOp>(loc, results[resultIdx], targets[resultIdx], loopVars);
            }
        };


        // Create loops from domain
        std::vector<Value> lbValues;
        std::vector<Value> ubValues;
        std::vector<int64_t> steps;
        for (ptrdiff_t boundIdx = 0; boundIdx < numDims; ++boundIdx) {
            auto lbAttr = rewriter.getIndexAttr(0);
            lbValues.push_back(rewriter.create<arith::ConstantOp>(loc, lbAttr));
            ubValues.push_back(*(op.getGridDim().begin() + boundIdx));
            steps.push_back(1);
        }

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

        if (m_makeParallelLoops) {
            auto parallelOp = rewriter.create<AffineParallelOp>(loc, resultTypes, reductions, affineMaps, lbValues, affineMaps, ubValues, steps);
            auto& parallelBlock = *parallelOp.getBody();
            auto loopVars = parallelBlock.getArguments();
            rewriter.setInsertionPointToStart(&parallelBlock);
            createLoopBody(rewriter, loc, loopVars);
        }
        else {
            buildAffineLoopNest(rewriter, loc, lbValues, ubValues, steps, createLoopBody);
        }

        rewriter.eraseOp(op);
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
    registry.insert<arith::ArithmeticDialect, AffineDialect, memref::MemRefDialect, linalg::LinalgDialect>();
}


void StencilToAffinePass::runOnOperation() {
    ConversionTarget target(getContext());
    target.addLegalDialect<AffineDialect>();
    target.addLegalDialect<arith::ArithmeticDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addIllegalDialect<stencil::StencilDialect>();
    target.addLegalOp<stencil::PrintOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<KernelLowering>(&getContext());
    patterns.add<KernelReturnLowering>(&getContext());
    patterns.add<KernelCallLowering>(&getContext(), 1, ArrayRef<StringRef>{}, m_makeParallelLoops);

    patterns.add<IndexLowering>(&getContext());
    patterns.add<JumpLowering>(&getContext());
    patterns.add<SampleLowering>(&getContext());
    patterns.add<JumpIndirectLowering>(&getContext());
    patterns.add<SampleIndirectLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}
