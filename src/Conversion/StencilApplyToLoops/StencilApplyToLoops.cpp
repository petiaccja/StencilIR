#include "StencilApplyToLoops.hpp"

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


struct ApplyOpLoweringBase : public OpRewritePattern<stencil::ApplyOp> {
    using OpRewritePattern<stencil::ApplyOp>::OpRewritePattern;

    static auto CreateLoopBody(stencil::ApplyOp op, OpBuilder& builder, Location loc, ValueRange loopVars) {
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

        auto call = builder.create<stencil::InvokeOp>(loc, resultTypes, op.getCallee(), offsetLoopVars, op.getInputs());

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
        rewriter.eraseOp(op);

        return success();
    }
};


void StencilApplyToLoopsPass::getDependentDialects(DialectRegistry& registry) const {
    registry.insert<arith::ArithmeticDialect,
                    func::FuncDialect,
                    memref::MemRefDialect,
                    vector::VectorDialect,
                    scf::SCFDialect,
                    stencil::StencilDialect>();
}


void StencilApplyToLoopsPass::runOnOperation() {
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithmeticDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<vector::VectorDialect>();
    target.addLegalDialect<stencil::StencilDialect>();

    target.addIllegalOp<stencil::ApplyOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<ApplyOpLoweringSCF>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}