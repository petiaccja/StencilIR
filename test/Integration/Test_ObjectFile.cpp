#include <IR/Types.hpp>
#include <Compiler/Pipelines.hpp>
#include <IR/ConvertOps.hpp>
#include <IR/Ops.hpp>
#include <Execution/Execution.hpp>

#include <filesystem>
#include <fstream>

#include <catch2/catch.hpp>


TEST_CASE("Object file", "[Program]") {
    using namespace dag;

    auto moduleOp = ModuleOp{};
    auto stencilir_add = moduleOp.Create<FuncOp>("stencilir_add",
                                               ast::FunctionType::Get({ ast::Int32, ast::Int32 },
                                                                      { ast::Int32 }),
                                               true);
    auto result = stencilir_add.Create<ArithmeticOp>(stencilir_add.GetRegionArg(0),
                                                     stencilir_add.GetRegionArg(1),
                                                     eArithmeticFunction::ADD)
                      .GetResult();
    stencilir_add.Create<ReturnOp>(std::vector{ result });

    mlir::MLIRContext context;
    auto convertedModule = mlir::dyn_cast<mlir::ModuleOp>(ConvertOperation(context, moduleOp));
    Compiler compiler{ TargetCPUPipeline(context) };
    auto compiledModule = compiler.Run(convertedModule);

    Runner runner{ compiledModule, 3 };
    auto objectFileBuffer = runner.GetObjectFile();
    auto objectFilePath = std::filesystem::temp_directory_path() / "test_object_file.obj";
    std::ofstream objectFile(objectFilePath, std::ios::binary | std::ios::trunc);
    objectFile.write(objectFileBuffer.data(), objectFileBuffer.size());
    objectFile.close();
    REQUIRE(std::filesystem::exists(objectFilePath));
}