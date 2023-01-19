#include <TestTools/FileCheck.hpp>

#include <Conversion/Passes.hpp>
#include <Diagnostics/Exception.hpp>
#include <Dialect/Stencil/Transforms/Passes.hpp>

#include <mlir/Dialect/Bufferization/Transforms/Passes.h>

#include <filesystem>
#include <iostream>

#include <catch2/catch.hpp>


static std::string TestFile(std::string_view name) {
    return (std::filesystem::path(FILE_CHECK_DIR) / name).string();
}


TEST_CASE("Convert stencil apply to loops", "[StencilDialect]") {
    REQUIRE(CheckFile(TestFile("ConvertStencilApplyToLoops.mlir"), Pass(createStencilApplyToLoopsPass())));
}


TEST_CASE("Convert stencil to standard", "[StencilDialect]") {
    REQUIRE(CheckFile(TestFile("ConvertStencilToStandard.mlir"), Pass(createStencilToStandardPass())));
}


TEST_CASE("Eliminate alloc tensors", "[StencilDialect]") {
    REQUIRE(CheckFile(TestFile("EliminateAllocTensors.mlir"), Pass(mlir::bufferization::createAllocTensorEliminationPass())));
}


TEST_CASE("Fuse apply ops", "[StencilDialect]") {
    REQUIRE(CheckFile(TestFile("FuseApplyOps.mlir"), Pass(createFuseApplyOpsPass())));
}