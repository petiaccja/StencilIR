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
#include <iostream>
#include <iterator>
#include <llvm/ADT/APFloat.h>
#include <memory>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
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

        std::vector<mlir::Type> functionParamTypes;
        std::vector<mlir::Type> targetParamTypes;
        const auto indexType = MemRefType::get({ 3 }, rewriter.getIndexType());
        functionParamTypes.push_back(indexType);
        for (auto& argument : op.getArgumentTypes()) {
            functionParamTypes.push_back(argument);
        }
        for (auto& result : op.getResultTypes()) {
            std::vector<int64_t> shape(2, ShapedType::kDynamicSize);
            std::vector<int64_t> strides(2, mlir::ShapedType::kDynamicStrideOrOffset);
            auto strideMap = mlir::makeStridedLinearLayoutMap(strides, 0, rewriter.getContext());
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
        std::vector<mlir::Value> targets;
        std::copy_n(blockArgs.rbegin(), op->getNumOperands(), std::back_inserter(targets));
        std::reverse(targets.begin(), targets.end());

        std::vector<mlir::Value> indices;
        for (size_t dimIdx = 0; dimIdx < 2; ++dimIdx) {
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

struct KernelCallLowering : public OpRewritePattern<mock::KernelCallOp> {
    using OpRewritePattern<mock::KernelCallOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mock::KernelCallOp op, PatternRewriter& rewriter) const override final {
        Location loc = op->getLoc();

        std::vector<Value> lbValues;
        std::vector<Value> ubValues;
        std::vector<int64_t> steps;
        for (size_t boundIdx = 0; boundIdx < op.getGridDim().size(); ++boundIdx) {
            auto lbAttr = rewriter.getIndexAttr(0);
            lbValues.push_back(rewriter.create<arith::ConstantOp>(loc, lbAttr));
            ubValues.push_back(*(op.getGridDim().begin() + boundIdx));
            steps.push_back(1);
        }

        buildAffineLoopNest(rewriter, loc, lbValues, ubValues, steps, [&op](OpBuilder& builder, Location loc, ValueRange loopVars) {
            auto MakeIndex = [&builder, &loc](int64_t value) {
                return builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(value));
            };

            std::vector<Value> operands;
            const Value indexPadding = MakeIndex(1);
            std::vector<Value> loopVarsExtended;
            for (const auto& loopVar : loopVars) {
                loopVarsExtended.push_back(loopVar);
            }
            while (loopVarsExtended.size() < 3) {
                loopVarsExtended.push_back(indexPadding);
            }

            auto index = builder.create<memref::AllocaOp>(loc, MemRefType::get({ 3 }, builder.getIndexType()));
            for (size_t i = 0; i < 3; ++i) {
                builder.create<memref::StoreOp>(loc, loopVarsExtended[i], index, ValueRange{ MakeIndex(i) });
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


void KernelToAffinePass::getDependentDialects(DialectRegistry& registry) const {
    registry.insert<arith::ArithmeticDialect, AffineDialect, memref::MemRefDialect>();
}


void KernelToAffinePass::runOnOperation() {
    ConversionTarget target(getContext());
    target.addLegalDialect<AffineDialect>();
    target.addLegalDialect<arith::ArithmeticDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addIllegalOp<mock::KernelCallOp>();
    target.addIllegalOp<mock::KernelReturnOp>();
    target.addIllegalOp<mock::KernelFuncOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<KernelFuncLowering>(&getContext());
    patterns.add<KernelReturnLowering>(&getContext());
    patterns.add<KernelCallLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}
