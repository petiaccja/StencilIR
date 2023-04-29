#include "StencilToLoops.hpp"

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
#include <iostream>
#include <vector>


namespace sir {

using namespace mlir;


struct GenerateOpLoweringBase : public OpRewritePattern<stencil::GenerateOp> {
    using OpRewritePattern<stencil::GenerateOp>::OpRewritePattern;

    static auto CreateLoopBody(stencil::GenerateOp op, OpBuilder& builder, Location loc, ValueRange loopVars) {
        auto ConstantIndex = [&builder, &loc](int64_t value) {
            return builder.create<arith::ConstantIndexOp>(loc, value);
        };

        // Get invocation result types from ApplyOp result tensors.
        std::vector<Type> resultTypes;
        for (const auto& type : op.getOutputs().getTypes()) {
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

        const size_t numDims = offsetLoopVars.size();
        mlir::Type indexType = VectorType::get({ int64_t(numDims) }, builder.getIndexType());
        Value index = builder.create<vector::SplatOp>(loc, indexType, ConstantIndex(0));
        for (size_t dimIdx = 0; dimIdx < numDims; ++dimIdx) {
            index = builder.create<vector::InsertElementOp>(loc, offsetLoopVars[dimIdx], index, ConstantIndex(dimIdx));
        }

        auto& generateRegion = op.getBody();
        auto execRegionOp = builder.create<mlir::scf::ExecuteRegionOp>(loc, resultTypes);
        mlir::IRMapping mapping;
        mapping.map(op->getRegion(0).front().getArgument(0), index);
        execRegionOp->getRegion(0).getBlocks().clear();
        generateRegion.cloneInto(&execRegionOp->getRegion(0), mapping);

        // Store return values to targets
        const auto results = execRegionOp->getResults();
        const auto outputs = op.getOutputs();
        const auto numResults = results.size();

        assert(results.size() == outputs.size());

        for (size_t resultIdx = 0; resultIdx < numResults; ++resultIdx) {
            builder.create<memref::StoreOp>(loc, results[resultIdx], outputs[resultIdx], loopVars);
        }
    };

    static int64_t GetDimension(stencil::GenerateOp op) {
        const auto& output0 = op.getOutputs()[0];
        const auto& type0 = output0.getType().cast<mlir::ShapedType>();
        return type0.getRank();
    }

    static auto GetOutputSize(stencil::GenerateOp op, OpBuilder& builder, Location loc)
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


struct GenerateOpLoweringParallel : GenerateOpLoweringBase {
    using GenerateOpLoweringBase::GenerateOpLoweringBase;

    LogicalResult matchAndRewrite(stencil::GenerateOp op, PatternRewriter& rewriter) const override final {
        Location loc = op->getLoc();
        const int64_t numDims = GetDimension(op);

        const auto ubValues = GetOutputSize(op, rewriter, loc);
        const auto lbValues = std::vector<Value>(numDims, rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0)));
        const std::vector<Value> steps(numDims, rewriter.create<arith::ConstantIndexOp>(loc, 1));

        rewriter.create<mlir::scf::ParallelOp>(loc, lbValues, ubValues, steps, [&op](OpBuilder& builder, Location loc, ValueRange loopVars) {
            CreateLoopBody(op, builder, loc, loopVars);
        });
        rewriter.eraseOp(op);

        return success();
    }
};


struct YieldOpLowering : public OpRewritePattern<stencil::YieldOp> {
    using OpRewritePattern<stencil::YieldOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(stencil::YieldOp op, PatternRewriter& rewriter) const override final {
        rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(op, op->getOperands());
        return success();
    }
};


void StencilToLoopsPass::getDependentDialects(DialectRegistry& registry) const {
    registry.insert<arith::ArithDialect,
                    func::FuncDialect,
                    memref::MemRefDialect,
                    vector::VectorDialect,
                    scf::SCFDialect,
                    stencil::StencilDialect>();
}


void StencilToLoopsPass::runOnOperation() {
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<vector::VectorDialect>();

    // It's never matching stencil.yield unless I do that first in a separate conversion pass.
    // I have not the slightest clue why...
    RewritePatternSet yieldPattern(&getContext());
    target.addIllegalOp<stencil::YieldOp>();
    yieldPattern.add<YieldOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(yieldPattern)))) {
        signalPassFailure();
    }

    RewritePatternSet generatePattern(&getContext());
    target.addIllegalOp<stencil::GenerateOp>();
    generatePattern.add<GenerateOpLoweringParallel>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(generatePattern)))) {
        signalPassFailure();
    }
}

} // namespace sir