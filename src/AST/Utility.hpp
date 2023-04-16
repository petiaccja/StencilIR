#include "Nodes.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/BuiltinTypes.h>

#include <optional>
#include <string>


mlir::Location ConvertLocation(mlir::OpBuilder& builder, const std::optional<ast::Location>& location);

std::string FormatType(mlir::Type type);

mlir::Type ConvertType(mlir::OpBuilder& builder, const ast::Type& type);

mlir::Value PromoteValue(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value value, mlir::Type type);

std::pair<mlir::Value, mlir::Value> PromoteToCommonType(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs);


bool CanImplicitCast(mlir::Type source, mlir::Type target);
mlir::Type CommonType(mlir::Type t, mlir::Type u);
std::optional<mlir::Value> Cast(mlir::Value value, mlir::Type type, mlir::OpBuilder& builder, mlir::Location loc);