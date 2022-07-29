#include "AllToLLVMPass.hpp"

#include <MockDialect/MockDialect.hpp>
#include <MockDialect/MockOps.hpp>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/Sequence.h>
#include <mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
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


void AllToLLVMPass::getDependentDialects(DialectRegistry& registry) const {
    registry.insert<arith::ArithmeticDialect, LLVM::LLVMDialect>();
}

void AllToLLVMPass::runOnOperation() {
    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();

    LLVMTypeConverter typeConverter(&getContext());

    RewritePatternSet patterns(&getContext());
    arith::populateArithmeticToLLVMConversionPatterns(typeConverter, patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);

    if (failed(applyFullConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}