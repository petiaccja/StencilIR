#include "StencilPrintToLLVM.hpp"

#include <Dialect/Stencil/IR/StencilOps.hpp>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/Sequence.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>


using namespace mlir;

class PrintOpLowering : public ConversionPattern {
public:
    explicit PrintOpLowering(MLIRContext* context) : ConversionPattern(stencil::PrintOp::getOperationName(), 1, context) {}

    LogicalResult matchAndRewrite(Operation* op,
                                  llvm::ArrayRef<Value> operands,
                                  ConversionPatternRewriter& rewriter) const override {
        const auto loc = op->getLoc();
        auto moduleOp = op->getParentOfType<ModuleOp>();
        auto printfRef = getOrInsertPrintf(rewriter, moduleOp);

        constexpr char fmt[] = "%f \0";
        Value formatSpecifierCst = getOrCreateGlobalString(loc,
                                                           rewriter,
                                                           "frmt_spec",
                                                           StringRef(fmt, strlen(fmt) + 1),
                                                           moduleOp);


        auto printOp = cast<stencil::PrintOp>(op);
        Value printArg = printOp.getInput();
        Value castPrintArg = rewriter.create<arith::ExtFOp>(loc, rewriter.getF64Type(), printArg);
        rewriter.create<LLVM::CallOp>(loc,
                                      ArrayRef<Type>(IntegerType::get(getContext(), 32)),
                                      printfRef.getValue(),
                                      ArrayRef<Value>({ formatSpecifierCst, castPrintArg }));
        rewriter.eraseOp(op);
        return success();
    }

private:
    static FlatSymbolRefAttr getOrInsertPrintf(ConversionPatternRewriter& rewriter, ModuleOp moduleOp) {
        MLIRContext* context = moduleOp->getContext();
        if (moduleOp.lookupSymbol<LLVM::LLVMFuncOp>("printf")) {
            return SymbolRefAttr::get(context, "printf");
        }

        auto llvmI32Ty = IntegerType::get(context, 32);
        auto llvmI8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
        auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy, true);

        PatternRewriter::InsertionGuard insertGuard(rewriter);
        rewriter.setInsertionPointToStart(moduleOp.getBody());
        rewriter.create<LLVM::LLVMFuncOp>(moduleOp.getLoc(), "printf", llvmFnType);
        return SymbolRefAttr::get(context, "printf");
    }

    static Value getOrCreateGlobalString(Location loc, OpBuilder& builder,
                                         StringRef name, StringRef value,
                                         ModuleOp moduleOp) {
        // Create the global at the entry of the module.
        LLVM::GlobalOp global;
        if (!(global = moduleOp.lookupSymbol<LLVM::GlobalOp>(name))) {
            OpBuilder::InsertionGuard insertGuard(builder);
            builder.setInsertionPointToStart(moduleOp.getBody());
            auto type = LLVM::LLVMArrayType::get(
                IntegerType::get(builder.getContext(), 8), value.size());
            global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                                    LLVM::Linkage::Internal, name,
                                                    builder.getStringAttr(value),
                                                    /*alignment=*/0);
        }

        // Get the pointer to the first character in the global string.
        Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
        Value cst0 = builder.create<LLVM::ConstantOp>(
            loc, IntegerType::get(builder.getContext(), 64),
            builder.getIntegerAttr(builder.getIndexType(), 0));
        return builder.create<LLVM::GEPOp>(
            loc,
            LLVM::LLVMPointerType::get(IntegerType::get(builder.getContext(), 8)),
            globalPtr, ArrayRef<Value>({ cst0, cst0 }));
    }
};


void StencilPrintToLLVMPass::getDependentDialects(DialectRegistry& registry) const {
    registry.insert<func::FuncDialect, LLVM::LLVMDialect, arith::ArithmeticDialect>();
}

void StencilPrintToLLVMPass::runOnOperation() {
    ConversionTarget target(getContext());
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<arith::ArithmeticDialect>();
    target.addIllegalOp<stencil::PrintOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<PrintOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}