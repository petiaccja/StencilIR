#include "ConvertOps.hpp"

#include "Converter.hpp"
#include "Ops.hpp"

#include <AST/Utility.hpp>
#include <Diagnostics/Exception.hpp>
#include <Diagnostics/Handlers.hpp>
#include <Dialect/Stencil/IR/StencilOps.hpp>

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Verifier.h>


namespace dag {


mlir::Location ConvertLocation(mlir::OpBuilder& builder, const std::optional<Location>& location) {
    if (location) {
        auto fileattr = builder.getStringAttr(location->file);
        return mlir::FileLineColLoc::get(fileattr, location->line, location->col);
    }
    return builder.getUnknownLoc();
}


//------------------------------------------------------------------------------
// Module, functions, stencils
//------------------------------------------------------------------------------

mlir::Operation* ConvertModuleOp(Converter& converter, OperationImpl& op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.Location());

    auto converted = builder.create<mlir::ModuleOp>(loc);

    builder.setInsertionPointToEnd(converted.getBody());
    for (const auto& op : op.Regions().front().operations) {
        converter(op);
    }

    return converted;
}


mlir::Operation* ConvertFuncOp(Converter& converter, OperationImpl& op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.Location());
    const auto& attr = std::any_cast<const FuncAttr&>(op.Attributes());

    auto functionType = ConvertType(builder, *attr.signature).dyn_cast<mlir::FunctionType>();

    auto converted = builder.create<mlir::func::FuncOp>(loc, attr.name, functionType);
    converter.MapEntryBlock(op.Regions().front(), converted.getBody().front());

    converted.setVisibility(attr.isPublic ? mlir::SymbolTable::Visibility::Public : mlir::SymbolTable::Visibility::Private);
    builder.setInsertionPointToEnd(&converted.getBody().front());
    for (const auto& op : op.Regions().front().operations) {
        converter(op);
    }

    return converted;
}


mlir::Operation* ConvertReturnOp(Converter& converter, OperationImpl& op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.Location());

    auto parentOp = builder.getInsertionBlock()->getParentOp();
    assert(parentOp);
    const bool isFuncOp = mlir::isa<mlir::func::FuncOp>(parentOp)
                          || parentOp->getParentOfType<mlir::func::FuncOp>();

    if (isFuncOp) {
        return builder.create<mlir::func::ReturnOp>(loc, operands);
    }
    else {
        return builder.create<stencil::ReturnOp>(loc, operands);
    }
}


//------------------------------------------------------------------------------
// Arithmetic-logic
//------------------------------------------------------------------------------

mlir::Operation* ConvertArithmeticOp(Converter& converter, OperationImpl& op, mlir::ValueRange operands) {
    auto& builder = converter.Builder();
    const auto loc = ConvertLocation(builder, op.Location());
    const auto& attr = std::any_cast<const eArithmeticFunction&>(op.Attributes());

    auto [lhs, rhs] = PromoteToCommonType(builder, loc, operands[0], operands[1]);
    bool isFloat = lhs.getType().isa<mlir::FloatType>();
    bool isUnsigned = lhs.getType().isUnsignedInteger();

    switch (attr) {
        case eArithmeticFunction::ADD:
            return isFloat ? builder.create<mlir::arith::AddFOp>(loc, lhs, rhs)
                           : builder.create<mlir::arith::AddIOp>(loc, lhs, rhs);
        case eArithmeticFunction::SUB:
            return isFloat ? builder.create<mlir::arith::SubFOp>(loc, lhs, rhs)
                           : builder.create<mlir::arith::SubIOp>(loc, lhs, rhs);
        case eArithmeticFunction::MUL:
            return isFloat ? builder.create<mlir::arith::MulFOp>(loc, lhs, rhs)
                           : builder.create<mlir::arith::MulIOp>(loc, lhs, rhs);
        case eArithmeticFunction::DIV:
            return isFloat      ? builder.create<mlir::arith::DivFOp>(loc, lhs, rhs)
                   : isUnsigned ? builder.create<mlir::arith::DivUIOp>(loc, lhs, rhs)
                                : builder.create<mlir::arith::DivSIOp>(loc, lhs, rhs);
        case eArithmeticFunction::MOD:
            return isFloat      ? builder.create<mlir::arith::RemFOp>(loc, lhs, rhs)
                   : isUnsigned ? builder.create<mlir::arith::RemUIOp>(loc, lhs, rhs)
                                : builder.create<mlir::arith::RemSIOp>(loc, lhs, rhs);
        case eArithmeticFunction::BIT_AND:
            return builder.create<mlir::arith::AndIOp>(loc, lhs, rhs);
        case eArithmeticFunction::BIT_OR:
            return builder.create<mlir::arith::OrIOp>(loc, lhs, rhs);
        case eArithmeticFunction::BIT_XOR:
            return builder.create<mlir::arith::XOrIOp>(loc, lhs, rhs);
        case eArithmeticFunction::BIT_SHL:
            return builder.create<mlir::arith::ShLIOp>(loc, lhs, rhs);
        case eArithmeticFunction::BIT_SHR:
            return isUnsigned ? builder.create<mlir::arith::ShRUIOp>(loc, lhs, rhs)
                              : builder.create<mlir::arith::ShRSIOp>(loc, lhs, rhs);
    }
    throw NotImplementedError("switch does not cover operator");
}



//------------------------------------------------------------------------------
// Generic
//------------------------------------------------------------------------------

mlir::Operation* ConvertOperation(mlir::MLIRContext& context, Operation op) {
    Converter converter{ context };

    converter.RegisterOp<ModuleOp>(&ConvertModuleOp);
    converter.RegisterOp<FuncOp>(&ConvertFuncOp);

    auto converted = converter(op);

    ScopedDiagnosticCollector diagnostics{ context };
    mlir::LogicalResult verificationResult = mlir::verify(converted);
    if (failed(verificationResult)) {
        converted->dump();
        throw CompilationError(diagnostics.TakeDiagnostics(), mlir::dyn_cast<mlir::ModuleOp>(converted));
    }
    return converted;
}


} // namespace dag