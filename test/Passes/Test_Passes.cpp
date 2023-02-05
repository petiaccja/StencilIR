#include <TestTools/FileCheck.hpp>

#include <Conversion/Passes.hpp>

#include <mlir/Dialect/Bufferization/Transforms/Passes.h>

#include <filesystem>

#include <catch2/catch.hpp>


static std::string TestFile(std::string_view name) {
    return (std::filesystem::path(FILE_CHECK_DIR) / name).string();
}


TEST_CASE("Convert stencil apply to loops", "[StencilDialect]") {
    REQUIRE(CheckFile(TestFile("ConvertStencilApplyToLoops.mlir"), createStencilApplyToLoopsPass()));
}


TEST_CASE("Convert stencil to standard", "[StencilDialect]") {
    REQUIRE(CheckFile(TestFile("ConvertStencilToStandard.mlir"), createStencilToStandardPass()));
}


TEST_CASE("Eliminate alloc tensors", "[StencilDialect]") {
    REQUIRE(CheckFile(TestFile("EliminateAllocTensors.mlir"), mlir::bufferization::createAllocTensorEliminationPass()));
}