#include "Types.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinTypes.h>

#include <optional>
#include <string>


namespace sir {


std::string FormatType(mlir::Type type);
mlir::Type ConvertType(mlir::OpBuilder& builder, const Type& type);
std::optional<mlir::Value> Cast(mlir::Value value, mlir::Type type, mlir::OpBuilder& builder, mlir::Location loc);


} // namespace sir