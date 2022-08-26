#include "StencilOpsToStandard.hpp"

#include <Dialect/Stencil/IR/StencilOps.hpp>

#include <llvm/ADT/ArrayRef.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/TypeRange.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

#include <algorithm>
#include <cassert>
#include <vector>


using namespace mlir;


struct JumpOpLowering : public OpRewritePattern<stencil::JumpOp> {
    using OpRewritePattern<stencil::JumpOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(stencil::JumpOp op, PatternRewriter& rewriter) const override final {
        Location loc = op->getLoc();

        auto ConstantIndex = [&rewriter, &loc](int64_t value) {
            return rewriter.create<arith::ConstantIndexOp>(loc, value);
        };

        Value index = op.getInputIndex();

        std::vector<int64_t> offsets;
        for (const auto& offset : op.getOffset().getAsRange<IntegerAttr>()) {
            offsets.push_back(offset.getInt());
        }
        const size_t numDims = offsets.size();

        const auto indexType = index.getType();
        Value offset = rewriter.create<vector::SplatOp>(loc, indexType, ConstantIndex(0));
        for (size_t dimIdx = 0; dimIdx < numDims; ++dimIdx) {
            offset = rewriter.create<vector::InsertElementOp>(loc, ConstantIndex(offsets[dimIdx]), offset, ConstantIndex(dimIdx));
        }

        Value result = rewriter.create<arith::AddIOp>(loc, index, offset);

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

        const auto indexType = index.getType().dyn_cast<VectorType>();
        const auto numDims = indexType.getShape()[0];

        std::vector<Value> indices;
        for (ptrdiff_t dimIdx = 0; dimIdx < numDims; ++dimIdx) {
            Value loadIndex = rewriter.create<arith::ConstantIndexOp>(loc, dimIdx);
            indices.push_back(rewriter.create<vector::ExtractElementOp>(loc, index, loadIndex));
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
        Value dimIndex = { rewriter.create<arith::ConstantIndexOp>(loc, dimension) };
        Value inputIndexElem = rewriter.create<vector::ExtractElementOp>(loc, inputIndex, dimIndex);

        auto map = op.getMap();
        auto mapElement = op.getMapElement();
        std::array<Value, 2> mapIndices = { inputIndexElem, mapElement };
        Value newIndexElem = rewriter.create<memref::LoadOp>(loc, map, mapIndices);

        Value outputIndex = rewriter.create<vector::InsertElementOp>(loc, newIndexElem, inputIndex, dimIndex);

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
        Value dimIndex = { rewriter.create<arith::ConstantIndexOp>(loc, dimension) };
        Value inputIndexElem = rewriter.create<vector::ExtractElementOp>(loc, inputIndex, dimIndex);

        auto field = op.getField();
        auto fieldElement = op.getFieldElement();
        std::array<Value, 2> mapIndices = { inputIndexElem, fieldElement };
        Value sample = rewriter.create<memref::LoadOp>(loc, field, mapIndices);

        rewriter.replaceOp(op, sample);

        return success();
    }
};


struct ForeachElementOpLowering : public OpRewritePattern<stencil::ForeachElementOp> {
    using OpRewritePattern<stencil::ForeachElementOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(stencil::ForeachElementOp op, PatternRewriter& rewriter) const override final {
        Location loc = op->getLoc();

        mlir::Value lb = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        mlir::Value ub = rewriter.create<memref::DimOp>(loc, op.getField(), op.getDim().getSExtValue());
        mlir::Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

        auto scfForOp = rewriter.create<scf::ForOp>(loc, lb, ub, step, op.getIterOperands());
        rewriter.eraseBlock(scfForOp.getBody());
        rewriter.inlineRegionBefore(op.getRegion(), scfForOp.getRegion(), scfForOp.getRegion().end());
        rewriter.replaceOp(op, scfForOp.getResults());

        return success();
    }
};


class YieldOpLowering : public OpRewritePattern<stencil::YieldOp> {
public:
    using OpRewritePattern<stencil::YieldOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(stencil::YieldOp op, PatternRewriter& rewriter) const override final {
        rewriter.replaceOpWithNewOp<scf::YieldOp>(op, op.getOperands());
        return success();
    }
};


void StencilOpsToStandardPass::getDependentDialects(DialectRegistry& registry) const {
    registry.insert<arith::ArithmeticDialect,
                    func::FuncDialect,
                    memref::MemRefDialect,
                    vector::VectorDialect,
                    scf::SCFDialect>();
}

void StencilOpsToStandardPass::runOnOperation() {
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithmeticDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<vector::VectorDialect>();
    target.addIllegalDialect<stencil::StencilDialect>();
    target.addLegalOp<stencil::PrintOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<JumpOpLowering>(&getContext());
    patterns.add<SampleOpLowering>(&getContext());
    patterns.add<JumpIndirectOpLowering>(&getContext());
    patterns.add<SampleIndirectOpLowering>(&getContext());
    patterns.add<ForeachElementOpLowering>(&getContext());
    patterns.add<YieldOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}