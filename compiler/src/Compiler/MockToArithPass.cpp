#include "MockToArithPass.hpp"

#include <MockDialect/MockDialect.hpp>
#include <MockDialect/MockOps.hpp>
#include <llvm/ADT/APFloat.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>


using namespace mlir;


struct ConstantOpLowering : public OpRewritePattern<mock::ConstantOp> {
    using OpRewritePattern<mock::ConstantOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mock::ConstantOp op, PatternRewriter& rewriter) const override final {
        const auto value = op.getValueAttr();
        Location loc = op->getLoc();
        auto lowered = rewriter.create<arith::ConstantOp>(loc, value);
        rewriter.replaceOp(op, Value(lowered));
        return success();
    }
};


void MockToArithPass::getDependentDialects(DialectRegistry& registry) const {
    registry.insert<arith::ArithmeticDialect>();
}


void MockToArithPass::runOnOperation() {
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithmeticDialect>();
    target.addIllegalDialect<mock::MockDialect>();
    target.addLegalOp<mock::PrintOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<ConstantOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}
