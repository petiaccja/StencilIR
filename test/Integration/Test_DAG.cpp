#include <DAG/ConvertOps.hpp>
#include <DAG/Ops.hpp>

#include <Compiler/Pipelines.hpp>
#include <Execution/Execution.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string_view>

#include <catch2/catch.hpp>

using namespace dag;

static ModuleOp CreateModule() {
    auto module = ModuleOp();

    {
        auto signature = ast::FunctionType::Get(
            { ast::IntegerType::Get(32, true), ast::IntegerType::Get(32, true) },
            { ast::IntegerType::Get(32, true) });
        auto func = FuncOp("main", signature);
        {
            auto lhs = func.Body().args[0];
            auto rhs = func.Body().args[1];
            auto add = ArithmeticOp(lhs, rhs, eArithmeticFunction::ADD);
            auto ret = ReturnOp({ add.Results()[0] });

            func.Body().operations.push_back(add);
            func.Body().operations.push_back(ret);
        }

        module.Body().operations.push_back(func);
    }

    return module;
}

TEST_CASE("Add", "[DAG]") {
    mlir::MLIRContext context;

    const auto mod = CreateModule();
    auto converted = mlir::dyn_cast<mlir::ModuleOp>(ConvertOperation(context, mod));

    Compiler compiler{ TargetCPUPipeline(context) };
    std::vector<StageResult> stages;

    mlir::ModuleOp compiled = compiler.Run(converted, stages);

    constexpr int optLevel = 3;
    Runner jitRunner{ compiled, optLevel };

    std::stringstream ss;
    for (auto& stage : stages) {
        ss << "// " << stage.name << std::endl;
        ss << stage.ir << "\n"
           << std::endl;
    }
    INFO(ss.str());

    jitRunner.Invoke("main", 1, 2);
}