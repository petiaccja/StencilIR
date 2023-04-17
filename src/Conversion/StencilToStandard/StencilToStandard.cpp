#include "StencilToStandard.hpp"

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
#include <numeric>
#include <vector>


namespace sir {


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


struct ProjectOpLowering : public OpRewritePattern<stencil::ProjectOp> {
    using OpRewritePattern<stencil::ProjectOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(stencil::ProjectOp op, PatternRewriter& rewriter) const override final {
        mlir::Value source = op.getSource();
        auto positions = op.getPositions();

        rewriter.replaceOpWithNewOp<vector::ShuffleOp>(op, source, source, positions);

        return success();
    }
};


struct ExtendOpLowering : public OpRewritePattern<stencil::ExtendOp> {
    using OpRewritePattern<stencil::ExtendOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(stencil::ExtendOp op, PatternRewriter& rewriter) const override final {
        mlir::Value source = op.getSource();
        mlir::VectorType sourceType = source.getType().dyn_cast<mlir::VectorType>();
        const auto size = sourceType.getShape()[0];
        assert(sourceType);
        mlir::SmallVector<int64_t, 8> mask(size);
        std::iota(mask.begin(), mask.end(), 0);
        mask.insert(mask.begin() + op.getPosition().getSExtValue(), size);

        std::array<int64_t, 1> shape = { 1 };
        mlir::Type valueType = mlir::VectorType::get(shape, rewriter.getIndexType());
        mlir::Value value = rewriter.create<vector::SplatOp>(op->getLoc(), valueType, op.getValue());

        rewriter.replaceOpWithNewOp<vector::ShuffleOp>(op, source, value, mask);

        return success();
    }
};


struct ExchangeOpLowering : public OpRewritePattern<stencil::ExchangeOp> {
    using OpRewritePattern<stencil::ExchangeOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(stencil::ExchangeOp op, PatternRewriter& rewriter) const override final {
        mlir::Value source = op.getSource();
        mlir::Value position = rewriter.create<mlir::arith::ConstantIndexOp>(op.getLoc(), op.getPosition().getSExtValue());
        mlir::Value value = op.getValue();

        rewriter.replaceOpWithNewOp<vector::InsertElementOp>(op, value, source, position);

        return success();
    }
};


struct ExtractOpLowering : public OpRewritePattern<stencil::ExtractOp> {
    using OpRewritePattern<stencil::ExtractOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(stencil::ExtractOp op, PatternRewriter& rewriter) const override final {
        mlir::Value source = op.getSource();
        mlir::Value position = rewriter.create<mlir::arith::ConstantIndexOp>(op.getLoc(), op.getPosition().getSExtValue());

        rewriter.replaceOpWithNewOp<vector::ExtractElementOp>(op, source, position);

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


void StencilToStandardPass::getDependentDialects(DialectRegistry& registry) const {
    registry.insert<arith::ArithmeticDialect,
                    func::FuncDialect,
                    memref::MemRefDialect,
                    vector::VectorDialect,
                    scf::SCFDialect>();
}

void StencilToStandardPass::runOnOperation() {
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
    patterns.add<ProjectOpLowering>(&getContext());
    patterns.add<ExtendOpLowering>(&getContext());
    patterns.add<ExchangeOpLowering>(&getContext());
    patterns.add<ExtractOpLowering>(&getContext());
    patterns.add<SampleOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}


} // namespace sir