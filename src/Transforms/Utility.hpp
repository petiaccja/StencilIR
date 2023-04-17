#pragma once

#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>


namespace sir {

mlir::FailureOr<mlir::Value> GetEquivalentBuffer(mlir::Value buffer, mlir::bufferization::AnalysisState& state);

}