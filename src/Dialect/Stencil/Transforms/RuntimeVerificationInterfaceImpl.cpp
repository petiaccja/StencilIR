#include <Dialect/Stencil/IR/StencilOps.hpp>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/Interfaces/RuntimeVerifiableOpInterface.h>


namespace stencil {

static mlir::StringAttr IndexUnderflowMessage(mlir::MLIRContext& context, int dim) {
    std::stringstream ss;
    ss << "index out of bounds: negative index for dimension #" << dim;
    return mlir::StringAttr::get(&context, ss.str());
}


static mlir::StringAttr IndexOverflowMessage(mlir::MLIRContext& context, int dim) {
    std::stringstream ss;
    ss << "index out of bounds: buffer size exceeded for dimension #" << dim;
    return mlir::StringAttr::get(&context, ss.str());
}


struct SampleOpInterface : public mlir::RuntimeVerifiableOpInterface::ExternalModel<SampleOpInterface, SampleOp> {
    void generateRuntimeVerification(mlir::Operation* op, mlir::OpBuilder& builder, mlir::Location loc) const {
        auto sampleOp = mlir::cast<SampleOp>(op);

        auto source = sampleOp.getField();
        auto index = sampleOp.getIndex();

        auto sourceType = mlir::cast<mlir::ShapedType>(source.getType());
        const bool isTensor = sourceType.isa<mlir::TensorType>();
        auto indexType = index.getType().cast<mlir::VectorType>();
        auto rank = sourceType.getRank();
        mlir::Value c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);

        for (int dim = 0; dim < rank; ++dim) {
            mlir::Value idx = builder.create<mlir::vector::ExtractOp>(
                loc, indexType.getElementType(), index, builder.getI64ArrayAttr({ dim }));
            mlir::Value cdim = builder.create<mlir::arith::ConstantIndexOp>(loc, dim);
            mlir::Value size = isTensor
                                   ? builder.create<mlir::tensor::DimOp>(loc, source, cdim).getResult()
                                   : builder.create<mlir::memref::DimOp>(loc, source, cdim).getResult();
            mlir::Value isNonnegative = builder.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::sle, c0, idx);
            mlir::Value isInBounds = builder.create<mlir::arith::CmpIOp>(
                loc, mlir::arith::CmpIPredicate::slt, idx, size);

            auto underflowMsg = IndexUnderflowMessage(*builder.getContext(), dim);
            builder.create<mlir::cf::AssertOp>(loc, isNonnegative, underflowMsg);
            auto overflowMsg = IndexOverflowMessage(*builder.getContext(), dim);
            builder.create<mlir::cf::AssertOp>(loc, isInBounds, overflowMsg);
        }
    }
};


void registerRuntimeVerifiableOpInterfaceExternalModels(mlir::DialectRegistry& registry) {
    registry.addExtension(+[](mlir::MLIRContext* ctx, stencil::StencilDialect* dialect) {
        SampleOp::attachInterface<SampleOpInterface>(*ctx);

        // Load additional dialects of which ops may get created.
        ctx->loadDialect<mlir::arith::ArithDialect,
                         mlir::cf::ControlFlowDialect,
                         mlir::tensor::TensorDialect,
                         mlir::memref::MemRefDialect,
                         mlir::vector::VectorDialect>();
    });
}


} // namespace stencil