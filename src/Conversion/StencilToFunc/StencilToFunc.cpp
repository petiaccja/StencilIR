#include "StencilToFunc.hpp"

#include <Dialect/Stencil/IR/StencilOps.hpp>

#include <llvm/ADT/ArrayRef.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
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


namespace sir {

using namespace mlir;



struct ApplyOpLowering : public OpRewritePattern<stencil::ApplyOp> {
    using OpRewritePattern<stencil::ApplyOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(stencil::ApplyOp op, PatternRewriter& rewriter) const override final {
        Location loc = op.getLoc();

        auto generateOp = rewriter.create<stencil::GenerateOp>(loc,
                                                               op->getResultTypes(),
                                                               op.getOutputs(),
                                                               op.getOffsets(),
                                                               op.getStaticOffsets());

        auto resultTypes = op->getResultTypes();
        mlir::SmallVector<mlir::Type> elementTypes;
        std::transform(resultTypes.begin(), resultTypes.end(), std::back_inserter(elementTypes), [](mlir::Type result) {
            auto shaped = result.dyn_cast<mlir::ShapedType>();
            assert(shaped);
            return shaped.getElementType();
        });

        auto& block = *generateOp.addEntryBlock();
        rewriter.setInsertionPointToEnd(&block);
        mlir::SmallVector<mlir::Value> args = { block.getArgument(0) };
        std::copy(op.getInputs().begin(), op.getInputs().end(), std::back_inserter(args));
        auto callOp = rewriter.create<func::CallOp>(loc, op.getCalleeAttr(), elementTypes, args);
        rewriter.create<stencil::YieldOp>(loc, callOp->getResults());

        rewriter.replaceOp(op, generateOp.getResults());
        return success();
    }
};


struct StencilOpLowering : public OpRewritePattern<stencil::StencilOp> {
    using OpRewritePattern<stencil::StencilOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(stencil::StencilOp op, PatternRewriter& rewriter) const override final {
        Location loc = op->getLoc();

        const int64_t numDims = op.getNumDimensions();

        std::vector<mlir::Type> functionParamTypes;
        const auto indexType = VectorType::get({ numDims }, rewriter.getIndexType());
        functionParamTypes.push_back(indexType);
        for (auto& argument : op.getArgumentTypes()) {
            functionParamTypes.push_back(argument);
        }
        auto functionReturnTypes = op.getResultTypes();
        auto functionType = rewriter.getFunctionType(functionParamTypes, functionReturnTypes);

        auto funcOp = rewriter.create<func::FuncOp>(loc, op.getSymName(), functionType);
        funcOp.setVisibility(SymbolTable::getSymbolVisibility(op));
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
        rewriter.replaceOpWithNewOp<func::ReturnOp>(op, op->getOperands());
        return success();
    }
};


struct InvokeOpLowering : public OpRewritePattern<stencil::InvokeOp> {
    using OpRewritePattern<stencil::InvokeOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(stencil::InvokeOp op, PatternRewriter& rewriter) const override {
        const auto index = op.getIndex();

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


void StencilToFuncPass::getDependentDialects(DialectRegistry& registry) const {
    registry.insert<arith::ArithDialect,
                    func::FuncDialect,
                    memref::MemRefDialect,
                    vector::VectorDialect,
                    scf::SCFDialect>();
}


void StencilToFuncPass::runOnOperation() {
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<vector::VectorDialect>();
    target.addLegalDialect<stencil::StencilDialect>();

    target.addIllegalOp<stencil::ApplyOp>();
    target.addIllegalOp<stencil::StencilOp>();
    target.addIllegalOp<stencil::ReturnOp>();
    target.addIllegalOp<stencil::InvokeOp>();
    target.addIllegalOp<stencil::IndexOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<ApplyOpLowering>(&getContext());
    patterns.add<StencilOpLowering>(&getContext());
    patterns.add<ReturnOpLowering>(&getContext());
    patterns.add<InvokeOpLowering>(&getContext());
    patterns.add<IndexOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

} // namespace sir