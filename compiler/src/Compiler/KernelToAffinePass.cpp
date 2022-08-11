#include "KernelToAffinePass.hpp"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include <MockDialect/MockDialect.hpp>
#include <MockDialect/MockOps.hpp>
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <llvm/ADT/APFloat.h>
#include <memory>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <vector>


using namespace mlir;

struct KernelFuncLowering : public OpRewritePattern<mock::KernelFuncOp> {
    using OpRewritePattern<mock::KernelFuncOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mock::KernelFuncOp op, PatternRewriter& rewriter) const override final {
        Location loc = op->getLoc();

        const int64_t numDims = op.getNumDimensions().getSExtValue();

        std::vector<mlir::Type> functionParamTypes;
        std::vector<mlir::Type> targetParamTypes;
        const auto indexType = MemRefType::get({ numDims }, rewriter.getIndexType());
        functionParamTypes.push_back(indexType);
        for (auto& argument : op.getArgumentTypes()) {
            functionParamTypes.push_back(argument);
        }
        for (auto& result : op.getResultTypes()) {
            constexpr auto offset = mlir::ShapedType::kDynamicStrideOrOffset;
            std::vector<int64_t> shape(numDims, ShapedType::kDynamicSize);
            std::vector<int64_t> strides(numDims, mlir::ShapedType::kDynamicStrideOrOffset);
            auto strideMap = mlir::makeStridedLinearLayoutMap(strides, offset, rewriter.getContext());
            Type type = MemRefType::get(shape, result, strideMap);
            targetParamTypes.push_back(type);
        }
        std::copy(targetParamTypes.begin(), targetParamTypes.end(), std::back_inserter(functionParamTypes));
        auto functionType = rewriter.getFunctionType(functionParamTypes, {});

        auto funcOp = rewriter.create<func::FuncOp>(loc, op.getSymName(), functionType);
        rewriter.inlineRegionBefore(op.getRegion(), funcOp.getBody(), funcOp.end());
        Block& block = funcOp.getBody().front();

        // insertArgument seems buggy with empty list.
        block.getNumArguments() == 0 ? block.addArgument(indexType, loc)
                                     : block.insertArgument(block.args_begin(), indexType, loc);
        for (auto& targetType : targetParamTypes) {
            block.addArgument(targetType, loc);
        }

        rewriter.eraseOp(op);
        return success();
    }
};

struct KernelReturnLowering : public OpRewritePattern<mock::KernelReturnOp> {
    using OpRewritePattern<mock::KernelReturnOp>::OpRewritePattern;

    LogicalResult match(mock::KernelReturnOp op) const override {
        auto parent = op->getParentOfType<func::FuncOp>();
        if (parent) {
            return success();
        }
        return failure();
    }

    void rewrite(mock::KernelReturnOp op, PatternRewriter& rewriter) const override {
        Location loc = op->getLoc();
        auto parent = op->getParentOfType<func::FuncOp>();
        assert(parent);

        const auto& blockArgs = parent.getBody().front().getArguments();
        const mlir::Value index = blockArgs.front();
        assert(index.getType().isa<mlir::MemRefType>());
        const auto indexType = index.getType().dyn_cast<mlir::MemRefType>();
        const int64_t numDims = indexType.getShape()[0];
        std::vector<mlir::Value> targets;
        std::copy_n(blockArgs.rbegin(), op->getNumOperands(), std::back_inserter(targets));
        std::reverse(targets.begin(), targets.end());

        std::vector<mlir::Value> indices;
        for (ptrdiff_t dimIdx = 0; dimIdx < numDims; ++dimIdx) {
            mlir::Value indexElement = rewriter.create<arith::ConstantIndexOp>(loc, dimIdx);
            indices.push_back(rewriter.create<memref::LoadOp>(loc, index, indexElement));
        }
        for (size_t targetIdx = 0; targetIdx < targets.size(); ++targetIdx) {
            auto target = targets[targetIdx];
            auto value = op->getOperand(targetIdx);
            rewriter.create<memref::StoreOp>(loc, value, target, indices);
        }

        rewriter.create<func::ReturnOp>(loc);
        rewriter.eraseOp(op);
    }
};

struct KernelCallLowering : public OpRewritePattern<mock::KernelLaunchOp> {
    using OpRewritePattern<mock::KernelLaunchOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mock::KernelLaunchOp op, PatternRewriter& rewriter) const override final {
        Location loc = op->getLoc();

        const int64_t numDims = op.getGridDim().size();

        std::vector<Value> lbValues;
        std::vector<Value> ubValues;
        std::vector<int64_t> steps;
        for (ptrdiff_t boundIdx = 0; boundIdx < numDims; ++boundIdx) {
            auto lbAttr = rewriter.getIndexAttr(0);
            lbValues.push_back(rewriter.create<arith::ConstantOp>(loc, lbAttr));
            ubValues.push_back(*(op.getGridDim().begin() + boundIdx));
            steps.push_back(1);
        }

        buildAffineLoopNest(rewriter, loc, lbValues, ubValues, steps, [&op, numDims](OpBuilder& builder, Location loc, ValueRange loopVars) {
            auto MakeIndex = [&builder, &loc](int64_t value) {
                return builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(value));
            };

            std::vector<Value> operands;
            auto index = builder.create<memref::AllocaOp>(loc, MemRefType::get({ numDims }, builder.getIndexType()));
            for (ptrdiff_t dimIdx = 0; dimIdx < numDims; ++dimIdx) {
                builder.create<memref::StoreOp>(loc, loopVars[dimIdx], index, ValueRange{ MakeIndex(dimIdx) });
            }

            operands.push_back(index);
            for (const auto& argument : op.getArguments()) {
                operands.push_back(argument);
            }
            for (const auto& target : op.getTargets()) {
                operands.push_back(target);
            }
            builder.create<func::CallOp>(loc, op.getCallee(), TypeRange{}, operands);
        });

        rewriter.eraseOp(op);
        return success();
    }
};

struct IndexLowering : public OpRewritePattern<mock::IndexOp> {
    using OpRewritePattern<mock::IndexOp>::OpRewritePattern;

    LogicalResult match(mock::IndexOp op) const override {
        auto parent = op->getParentOfType<func::FuncOp>();
        if (parent) {
            return success();
        }
        return failure();
    }

    void rewrite(mock::IndexOp op, PatternRewriter& rewriter) const override {
        auto parent = op->getParentOfType<func::FuncOp>();
        assert(parent);

        const auto& blockArgs = parent.getBody().front().getArguments();
        const mlir::Value index = blockArgs.front();
        rewriter.replaceOp(op, { index });
    }
};

struct OffsetLowering : public OpRewritePattern<mock::OffsetOp> {
    using OpRewritePattern<mock::OffsetOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mock::OffsetOp op, PatternRewriter& rewriter) const override final {
        Location loc = op->getLoc();

        Value index = op.getIndex();
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

struct SampleLowering : public OpRewritePattern<mock::SampleOp> {
    using OpRewritePattern<mock::SampleOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mock::SampleOp op, PatternRewriter& rewriter) const override final {
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


void KernelToAffinePass::getDependentDialects(DialectRegistry& registry) const {
    registry.insert<arith::ArithmeticDialect, AffineDialect, memref::MemRefDialect, linalg::LinalgDialect>();
}


void KernelToAffinePass::runOnOperation() {
    ConversionTarget target(getContext());
    target.addLegalDialect<AffineDialect>();
    target.addLegalDialect<arith::ArithmeticDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addIllegalDialect<mock::MockDialect>();
    target.addLegalOp<mock::PrintOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<KernelFuncLowering>(&getContext());
    patterns.add<KernelReturnLowering>(&getContext());
    patterns.add<KernelCallLowering>(&getContext());

    patterns.add<IndexLowering>(&getContext());
    patterns.add<OffsetLowering>(&getContext());
    patterns.add<SampleLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}
