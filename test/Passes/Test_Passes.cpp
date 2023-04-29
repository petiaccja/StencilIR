#include <TestTools/FileCheck.hpp>

#include <Conversion/Passes.hpp>
#include <Diagnostics/Exception.hpp>
#include <Dialect/Stencil/Transforms/Passes.hpp>
#include <Transforms/Passes.hpp>

#include <mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Transforms/Passes.h>

#include <filesystem>
#include <iostream>

#include <catch2/catch.hpp>


using namespace sir;


static std::string TestFile(std::string_view name) {
    return (std::filesystem::path(FILE_CHECK_DIR) / name).string();
}


TEST_CASE("Convert stencil to loops", "[StencilDialect]") {
    REQUIRE(CheckFile(TestFile("ConvertStencilToLoops.mlir"), Pass(createStencilToLoopsPass())));
}


TEST_CASE("Convert stencil to standard", "[StencilDialect]") {
    REQUIRE(CheckFile(TestFile("ConvertStencilToStandard.mlir"), Pass(createStencilToStandardPass())));
}


TEST_CASE("Eliminate alloc tensors", "[StencilDialect]") {
    REQUIRE(CheckFile(TestFile("EliminateAllocTensors.mlir"), Pass(mlir::bufferization::createEmptyTensorEliminationPass())));
}


TEST_CASE("Inline stencil invocations", "[StencilDialect]") {
    REQUIRE(CheckFile(TestFile("InlineStencilInvocations.mlir"), Pass(mlir::createInlinerPass())));
}


TEST_CASE("Fuse apply ops", "[StencilDialect]") {
    REQUIRE(CheckFile(TestFile("FuseApplyOps.mlir"), Pass(createFuseApplyOpsPass())));
}


TEST_CASE("Fuse extract slice ops", "[StencilDialect]") {
    REQUIRE(CheckFile(TestFile("FuseExtractSliceOps.mlir"), Pass(createFuseExtractSliceOpsPass())));
}


TEST_CASE("Deduplicate apply op inputs", "[StencilDialect]") {
    REQUIRE(CheckFile(TestFile("DeduplicateApplyInputs.mlir"), Pass(createDeduplicateApplyInputsPass())));
}


TEST_CASE("Reduce dim ops", "[Canonicalization]") {
    REQUIRE(CheckFile(TestFile("ReduceDimOps.mlir"), Pass(createReduceDimOpsPass())));
}


TEST_CASE("Eliminate slicing", "[Canonicalization]") {
    REQUIRE(CheckFile(TestFile("EliminateSlicing.mlir"), Pass(createEliminateSlicingPass())));
}


TEST_CASE("Convert stencil to func", "[StencilDialect]") {
    REQUIRE(CheckFile(TestFile("ConvertStencilToFunc.mlir"), Pass(createStencilToFuncPass())));
}


TEST_CASE("Regression: bufferize crash", "[StencilDialect]") {
    mlir::MLIRContext context;
    mlir::bufferization::OneShotBufferizationOptions bufferizationOptions;
    bufferizationOptions.allowUnknownOps = false;
    bufferizationOptions.allowReturnAllocs = false;
    bufferizationOptions.createDeallocs = true;
    bufferizationOptions.defaultMemorySpace = mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 64), 0);
    bufferizationOptions.functionBoundaryTypeConversion = mlir::bufferization::LayoutMapOption::FullyDynamicLayoutMap;
    bufferizationOptions.bufferizeFunctionBoundaries = true;
    REQUIRE(CheckFile(context,
                      TestFile("Regression_BufferizeCrash.mlir"),
                      Pass(mlir::bufferization::createEmptyTensorToAllocTensorPass()),
                      Pass(mlir::bufferization::createOneShotBufferizePass(bufferizationOptions))));
}
