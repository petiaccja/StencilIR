#include "Utility.hpp"

#include "Nodes.hpp"

#include <Diagnostics/Exception.hpp>

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinTypes.h>

#include <optional>
#include <string>



mlir::Location ConvertLocation(mlir::OpBuilder& builder, const std::optional<ast::Location>& location) {
    if (location) {
        auto fileattr = builder.getStringAttr(location->file);
        return mlir::FileLineColLoc::get(fileattr, location->line, location->col);
    }
    return builder.getUnknownLoc();
}

std::string FormatType(mlir::Type type) {
    std::string s;
    llvm::raw_string_ostream os{ s };
    type.print(os);
    return s;
}


mlir::Type ConvertType(mlir::OpBuilder& builder, const ast::Type& type) {
    if (auto integerType = dynamic_cast<const ast::IntegerType*>(&type)) {
        if (!integerType->isSigned) {
            throw std::invalid_argument("unsigned types are not supported due to arith.constant behavior; TODO: add support");
        }
        return builder.getIntegerType(integerType->size);
    }
    else if (auto floatType = dynamic_cast<const ast::FloatType*>(&type)) {
        switch (floatType->size) {
            case 16: return builder.getF16Type();
            case 32: return builder.getF32Type();
            case 64: return builder.getF64Type();
        }
        throw std::invalid_argument("only 16, 32, and 64-bit floats are supported");
    }
    else if (auto indexType = dynamic_cast<const ast::IndexType*>(&type)) {
        return builder.getIndexType();
    }
    else if (auto fieldType = dynamic_cast<const ast::FieldType*>(&type)) {
        const mlir::Type elementType = ConvertType(builder, *fieldType->elementType);
        std::vector<int64_t> shape(fieldType->numDimensions, mlir::ShapedType::kDynamicSize);
        return mlir::RankedTensorType::get(shape, elementType);
    }
    else if (auto functionType = dynamic_cast<const ast::FunctionType*>(&type)) {
        mlir::SmallVector<mlir::Type> parameters;
        mlir::SmallVector<mlir::Type> results;
        std::ranges::transform(functionType->parameters, std::back_inserter(parameters), [&](const auto& type) {
            return ConvertType(builder, *type);
        });
        std::ranges::transform(functionType->results, std::back_inserter(results), [&](const auto& type) {
            return ConvertType(builder, *type);
        });
        return builder.getFunctionType(parameters, results);
    }
    else {
        std::stringstream ss;
        ss << "could not convert type \"" << type << "\" to MLIR type";
        throw std::invalid_argument(ss.str());
    }
}


mlir::Value PromoteValue(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value value, mlir::Type type) {
    auto inputType = value.getType();
    if (inputType.isa<mlir::IntegerType>() && type.isa<mlir::IntegerType>()) {
        auto inputIntType = inputType.dyn_cast<mlir::IntegerType>();
        auto intType = type.dyn_cast<mlir::IntegerType>();
        if (inputIntType.getSignedness() == intType.getSignedness()) {
            if (inputIntType.getWidth() == intType.getWidth()) {
                return value;
            }
            if (inputIntType.getWidth() < intType.getWidth()) {
                const auto signedness = inputIntType.getSignedness();
                if (signedness == mlir::IntegerType::Unsigned) {
                    return builder.create<mlir::arith::ExtUIOp>(loc, type, value);
                }
                return builder.create<mlir::arith::ExtSIOp>(loc, type, value);
            }
        }
        return nullptr;
    }
    if (inputType.isa<mlir::FloatType>() && type.isa<mlir::FloatType>()) {
        auto inputFloatType = inputType.dyn_cast<mlir::FloatType>();
        auto floatType = type.dyn_cast<mlir::FloatType>();
        if (inputFloatType.getWidth() == floatType.getWidth()) {
            return value;
        }
        if (inputFloatType.getWidth() < floatType.getWidth()) {
            return builder.create<mlir::arith::ExtFOp>(loc, type, value);
        }
        return nullptr;
    }
    if (inputType.isa<mlir::IntegerType>() && type.isa<mlir::FloatType>()) {
        auto inputIntType = inputType.dyn_cast<mlir::IntegerType>();
        auto floatType = type.dyn_cast<mlir::FloatType>();
        if (inputIntType.getWidth() < floatType.getFPMantissaWidth()) {
            if (inputIntType.getSignedness() == mlir::IntegerType::Unsigned) {
                return builder.create<mlir::arith::UIToFPOp>(loc, floatType, value);
            }
            return builder.create<mlir::arith::SIToFPOp>(loc, floatType, value);
        }
        return nullptr;
    }
    return nullptr;
}


std::pair<mlir::Value, mlir::Value> PromoteToCommonType(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs) {
    const auto lhsType = lhs.getType();
    const auto rhsType = rhs.getType();
    if (lhsType == rhsType) {
        return { lhs, rhs };
    }
    if (auto promotedLhs = PromoteValue(builder, loc, lhs, rhsType)) {
        return { promotedLhs, rhs };
    }
    if (auto promotedRhs = PromoteValue(builder, loc, rhs, lhsType)) {
        return { lhs, promotedRhs };
    }
    throw OperandTypeError(loc, { FormatType(lhsType), FormatType(rhsType) });
}