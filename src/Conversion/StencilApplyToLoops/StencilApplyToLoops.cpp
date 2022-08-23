#include "StencilApplyToLoops.hpp"

#include <StencilDialect/StencilDialect.hpp>
#include <StencilDialect/StencilOps.hpp>

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

struct StencilOpLowering : public OpRewritePattern<stencil::StencilOp> {
    using OpRewritePattern<stencil::StencilOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(stencil::StencilOp op, PatternRewriter& rewriter) const override final {
        Location loc = op->getLoc();

        const int64_t numDims = op.getNumDimensions().getSExtValue();

        std::vector<mlir::Type> functionParamTypes;
        const auto indexType = VectorType::get({ numDims }, rewriter.getIndexType());
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

struct ApplyOpLoweringBase : public OpRewritePattern<stencil::ApplyOp> {
    using OpRewritePattern<stencil::ApplyOp>::OpRewritePattern;

    static auto CreateLoopBody(stencil::ApplyOp op, OpBuilder& builder, Location loc, ValueRange loopVars) {
        // Get invocation result types from ApplyOp result tensors.
        std::vector<Type> resultTypes;
        for (const auto& type : op->getResultTypes()) {
            mlir::ShapedType shapedType = type.dyn_cast<ShapedType>();
            assert(shapedType);
            resultTypes.push_back(shapedType.getElementType());
        }

        // Offset loopVars
        std::vector<Value> offsetLoopVars;
        if (!op.getOffsets().empty()) {
            auto offsetIt = op.getOffsets().begin();
            for (const auto& loopVar : loopVars) {
                offsetLoopVars.push_back(builder.create<arith::AddIOp>(loc, loopVar, *offsetIt++));
            }
        }
        else if (!op.getStaticOffsets().empty()) {
            auto offsetIt = op.getStaticOffsets().getAsRange<IntegerAttr>().begin();
            for (const auto& loopVar : loopVars) {
                auto attr = *offsetIt++;
                Value staticOffset = builder.create<arith::ConstantIndexOp>(loc, attr.getInt());
                offsetLoopVars.push_back(builder.create<arith::AddIOp>(loc, loopVar, staticOffset));
            }
        }
        else {
            for (const auto& loopVar : loopVars) {
                offsetLoopVars.push_back(loopVar);
            }
        }

        auto call = builder.create<stencil::InvokeStencilOp>(loc, resultTypes, op.getCallee(), offsetLoopVars, op.getInputs());

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

struct ApplyOpLoweringSCF : ApplyOpLoweringBase {
    using ApplyOpLoweringBase::ApplyOpLoweringBase;

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


struct InvokeStencilLowering : public OpRewritePattern<stencil::InvokeStencilOp> {
    using OpRewritePattern<stencil::InvokeStencilOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(stencil::InvokeStencilOp op, PatternRewriter& rewriter) const override {
        Location loc = op->getLoc();

        auto ConstantIndex = [&rewriter, &loc](int64_t value) {
            return rewriter.create<arith::ConstantIndexOp>(loc, value);
        };

        const auto indices = op.getIndices();
        const size_t numDims = indices.size();

        mlir::Type indexType = VectorType::get({ int64_t(numDims) }, rewriter.getIndexType());
        Value index = rewriter.create<vector::SplatOp>(loc, indexType, ConstantIndex(0));
        for (size_t dimIdx = 0; dimIdx < numDims; ++dimIdx) {
            index = rewriter.create<vector::InsertElementOp>(loc, indices[dimIdx], index, ConstantIndex(dimIdx));
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


void StencilApplyToLoopsPass::getDependentDialects(DialectRegistry& registry) const {
    registry.insert<arith::ArithmeticDialect,
                    func::FuncDialect,
                    memref::MemRefDialect,
                    vector::VectorDialect,
                    scf::SCFDialect>();
}


void StencilApplyToLoopsPass::runOnOperation() {
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithmeticDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<vector::VectorDialect>();

    target.addIllegalOp<stencil::StencilOp>();
    target.addIllegalOp<stencil::ReturnOp>();
    target.addIllegalOp<stencil::InvokeStencilOp>();
    target.addIllegalOp<stencil::ApplyOp>();
    target.addIllegalOp<stencil::IndexOp>();

    target.addLegalOp<stencil::PrintOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<StencilOpLowering>(&getContext());
    patterns.add<ReturnOpLowering>(&getContext());
    patterns.add<InvokeStencilLowering>(&getContext());
    patterns.add<ApplyOpLoweringSCF>(&getContext());
    patterns.add<IndexOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}