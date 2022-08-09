#include "AllToLLVMPass.hpp"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include <MockDialect/MockDialect.hpp>
#include <MockDialect/MockOps.hpp>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/Sequence.h>
#include <mlir/Conversion/AffineToStandard/AffineToStandard.h>
#include <mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/TensorToLinalg/TensorToLinalg.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
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

void PrepareToLLVMPass::getDependentDialects(DialectRegistry& registry) const {
    registry.insert<arith::ArithmeticDialect,
                    func::FuncDialect,
                    tensor::TensorDialect,
                    AffineDialect,
                    linalg::LinalgDialect,
                    scf::SCFDialect,
                    cf::ControlFlowDialect,
                    LLVM::LLVMDialect>();
}

void PrepareToLLVMPass::runOnOperation() {
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithmeticDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<cf::ControlFlowDialect>();

    RewritePatternSet patterns(&getContext());
    populateTensorToLinalgPatterns(patterns);
    populateAffineToStdConversionPatterns(patterns);
    populateSCFToControlFlowConversionPatterns(patterns);


    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

void AllToLLVMPass::getDependentDialects(DialectRegistry& registry) const {
    registry.insert<arith::ArithmeticDialect,
                    func::FuncDialect,
                    tensor::TensorDialect,
                    AffineDialect,
                    linalg::LinalgDialect,
                    scf::SCFDialect,
                    cf::ControlFlowDialect,
                    memref::MemRefDialect,
                    LLVM::LLVMDialect>();
}

void AllToLLVMPass::runOnOperation() {
    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();

    LLVMTypeConverter typeConverter(&getContext());

    RewritePatternSet patterns(&getContext());
    populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
    populateLinalgToLLVMConversionPatterns(typeConverter, patterns);
    arith::populateArithmeticToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);

    if (failed(applyFullConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}