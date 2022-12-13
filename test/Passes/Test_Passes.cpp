#include <TestTools/FileCheck.hpp>

#include <Conversion/Passes.hpp>

#include <filesystem>

#include <catch2/catch.hpp>


static std::string TestFile(std::string_view name) {
    return std::filesystem::path(FILE_CHECK_DIR) / name;
}


TEST_CASE("Convert stencil apply to loops", "[StencilDialect]") {
    REQUIRE(CheckFile(TestFile("ConvertStencilApplyToLoops.mlir"), createStencilApplyToLoopsPass()));
}


TEST_CASE("Convert stencil to standard", "[StencilDialect]") {
    REQUIRE(CheckFile(TestFile("ConvertStencilToStandard.mlir"), createStencilToStandardPass()));
}
