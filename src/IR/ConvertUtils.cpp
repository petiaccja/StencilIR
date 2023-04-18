#include "ConvertUtils.hpp"

#include <Diagnostics/Exception.hpp>

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinTypes.h>

#include <optional>
#include <string>



namespace sir {


std::string FormatType(mlir::Type type) {
    std::string s;
    llvm::raw_string_ostream os{ s };
    type.print(os);
    return s;
}


mlir::Type ConvertType(mlir::OpBuilder& builder, const Type& type) {
    if (auto integerType = dynamic_cast<const IntegerType*>(&type)) {
        if (!integerType->isSigned) {
            throw std::invalid_argument("unsigned types are not supported due to arith.constant behavior; TODO: add support");
        }
        return builder.getIntegerType(integerType->size);
    }
    else if (auto floatType = dynamic_cast<const FloatType*>(&type)) {
        switch (floatType->size) {
            case 16: return builder.getF16Type();
            case 32: return builder.getF32Type();
            case 64: return builder.getF64Type();
        }
        throw std::invalid_argument("only 16, 32, and 64-bit floats are supported");
    }
    else if (auto indexType = dynamic_cast<const IndexType*>(&type)) {
        return builder.getIndexType();
    }
    else if (auto fieldType = dynamic_cast<const FieldType*>(&type)) {
        const mlir::Type elementType = ConvertType(builder, *fieldType->elementType);
        std::vector<int64_t> shape(fieldType->numDimensions, mlir::ShapedType::kDynamicSize);
        return mlir::RankedTensorType::get(shape, elementType);
    }
    else if (auto functionType = dynamic_cast<const FunctionType*>(&type)) {
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


mlir::Type MakeSignedType(mlir::IntegerType type) {
    return mlir::IntegerType::get(type.getContext(), type.getWidth(), mlir::IntegerType::SignednessSemantics::Signed);
}


mlir::Type MakeUnsignedType(mlir::IntegerType type) {
    return mlir::IntegerType::get(type.getContext(), type.getWidth(), mlir::IntegerType::SignednessSemantics::Unsigned);
}


namespace cast_methods {

    mlir::Value CastInt2Int(mlir::IntegerType sourceType,
                            mlir::IntegerType targetType,
                            mlir::Value value,
                            mlir::OpBuilder& builder,
                            mlir::Location loc) {
        // Extension
        if (sourceType.getWidth() < targetType.getWidth()) {
            return !targetType.isUnsigned()
                       ? builder.create<mlir::arith::ExtSIOp>(loc, targetType, value).getResult()
                       : builder.create<mlir::arith::ExtUIOp>(loc, targetType, value).getResult();
        }
        // Truncation
        if (sourceType.getWidth() > targetType.getWidth()) {
            return builder.create<mlir::arith::TruncIOp>(loc, targetType, value).getResult();
        }
        // Neither
        return value;
    }

    mlir::Value CastInt2Float(mlir::IntegerType sourceType,
                              mlir::FloatType targetType,
                              mlir::Value value,
                              mlir::OpBuilder& builder,
                              mlir::Location loc) {
        return !sourceType.isUnsigned()
                   ? builder.create<mlir::arith::SIToFPOp>(loc, targetType, value).getResult()
                   : builder.create<mlir::arith::UIToFPOp>(loc, targetType, value).getResult();
    }

    mlir::Value CastInt2Index(mlir::IntegerType sourceType,
                              mlir::IndexType targetType,
                              mlir::Value value,
                              mlir::OpBuilder& builder,
                              mlir::Location loc) {
        return builder.create<mlir::arith::IndexCastOp>(loc, targetType, value).getResult();
    }

    mlir::Value CastFloat2Float(mlir::FloatType sourceType,
                                mlir::FloatType targetType,
                                mlir::Value value,
                                mlir::OpBuilder& builder,
                                mlir::Location loc) {
        // Extension
        if (sourceType.getWidth() < targetType.getWidth()) {
            return builder.create<mlir::arith::ExtFOp>(loc, targetType, value).getResult();
        }
        // Truncation
        if (sourceType.getWidth() > targetType.getWidth()) {
            return builder.create<mlir::arith::TruncFOp>(loc, targetType, value).getResult();
        }
        // Neither
        return value;
    }

    mlir::Value CastFloat2Int(mlir::FloatType sourceType,
                              mlir::IntegerType targetType,
                              mlir::Value value,
                              mlir::OpBuilder& builder,
                              mlir::Location loc) {
        return !targetType.isUnsigned()
                   ? builder.create<mlir::arith::FPToSIOp>(loc, targetType, value).getResult()
                   : builder.create<mlir::arith::FPToUIOp>(loc, targetType, value).getResult();
    }

    mlir::Value CastFloat2Index(mlir::FloatType sourceType,
                                mlir::IndexType targetType,
                                mlir::Value value,
                                mlir::OpBuilder& builder,
                                mlir::Location loc) {
        auto intermediateType = builder.getIntegerType(64, true);
        auto intValue = CastFloat2Int(sourceType, intermediateType, value, builder, loc);
        auto indexValue = CastInt2Index(intermediateType, targetType, intValue, builder, loc);
        return indexValue;
    }

    mlir::Value CastIndex2Int(mlir::IndexType sourceType,
                              mlir::IntegerType targetType,
                              mlir::Value value,
                              mlir::OpBuilder& builder,
                              mlir::Location loc) {
        return builder.create<mlir::arith::IndexCastOp>(loc, targetType, value).getResult();
    }

    mlir::Value CastIndex2Float(mlir::IndexType sourceType,
                                mlir::FloatType targetType,
                                mlir::Value value,
                                mlir::OpBuilder& builder,
                                mlir::Location loc) {
        auto intermediateType = builder.getIntegerType(64, true);
        auto intValue = CastIndex2Int(sourceType, intermediateType, value, builder, loc);
        auto floatValue = CastInt2Float(intermediateType, targetType, intValue, builder, loc);
        return floatValue;
    }

    mlir::Value CastIndex2Index(mlir::IndexType sourceType,
                                mlir::IndexType targetType,
                                mlir::Value value,
                                mlir::OpBuilder& builder,
                                mlir::Location loc) {
        return value;
    }

} // namespace cast_methods

template <class Source, class Target>
std::optional<mlir::Value> TryCast(mlir::Value (*func)(Source, Target, mlir::Value, mlir::OpBuilder&, mlir::Location),
                                   mlir::Type sourceType,
                                   mlir::Type targetType,
                                   mlir::Value value,
                                   mlir::OpBuilder& builder,
                                   mlir::Location loc) {
    auto source = sourceType.dyn_cast<Source>();
    auto target = targetType.dyn_cast<Target>();
    if (source && target) {
        return func(source, target, value, builder, loc);
    }
    return {};
}

std::optional<mlir::Value> Cast(mlir::Value value, mlir::Type target, mlir::OpBuilder& builder, mlir::Location loc) {
    auto source = value.getType();

    // Int to ...
    if (auto cast = TryCast(&cast_methods::CastInt2Int, source, target, value, builder, loc)) {
        return cast;
    }
    if (auto cast = TryCast(&cast_methods::CastInt2Float, source, target, value, builder, loc)) {
        return cast;
    }
    if (auto cast = TryCast(&cast_methods::CastInt2Index, source, target, value, builder, loc)) {
        return cast;
    }

    // Float to ...
    if (auto cast = TryCast(&cast_methods::CastFloat2Float, source, target, value, builder, loc)) {
        return cast;
    }
    if (auto cast = TryCast(&cast_methods::CastFloat2Int, source, target, value, builder, loc)) {
        return cast;
    }
    if (auto cast = TryCast(&cast_methods::CastFloat2Index, source, target, value, builder, loc)) {
        return cast;
    }

    // Index to ...
    if (auto cast = TryCast(&cast_methods::CastIndex2Float, source, target, value, builder, loc)) {
        return cast;
    }
    if (auto cast = TryCast(&cast_methods::CastIndex2Int, source, target, value, builder, loc)) {
        return cast;
    }
    if (auto cast = TryCast(&cast_methods::CastIndex2Index, source, target, value, builder, loc)) {
        return cast;
    }

    return {};
}


} // namespace sir