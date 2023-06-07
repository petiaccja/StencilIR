#include "Utility/RunModule.hpp"

#include <Compiler/Pipelines.hpp>
#include <Execution/Execution.hpp>
#include <IR/ConvertOps.hpp>
#include <IR/Ops.hpp>
#include <IR/Types.hpp>

#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>

#include <catch2/catch.hpp>

using namespace sir;


auto CreateModule() {
    const auto scalarType = Float32;
    const auto fieldType = FieldType::Get(Float32, 1);

    auto mod = ops::ModuleOp{};

    auto helper_fun = mod.Create<ops::FuncOp>("helper_fun",
                                              FunctionType::Get({ scalarType, scalarType },
                                                                { scalarType }),
                                              false);
    auto result = helper_fun.Create<ops::ArithmeticOp>(helper_fun.GetRegionArg(0),
                                                       helper_fun.GetRegionArg(1),
                                                       ops::eArithmeticFunction::ADD)
                      .GetResult();
    helper_fun.Create<ops::ReturnOp>(std::vector{ result });

    auto kernel_fun = mod.Create<ops::StencilOp>("kernel_fun",
                                                 FunctionType::Get({ fieldType,
                                                                     fieldType },
                                                                   { scalarType }),
                                                 1,
                                                 false);
    auto index = kernel_fun.Create<ops::IndexOp>().GetResult();
    auto s1 = kernel_fun.Create<ops::SampleOp>(kernel_fun.GetRegionArg(0), index).GetResult();
    auto s2 = kernel_fun.Create<ops::SampleOp>(kernel_fun.GetRegionArg(1), index).GetResult();
    auto r = kernel_fun.Create<ops::CallOp>(helper_fun, std::vector{ s1, s2 }).GetResults()[0];
    kernel_fun.Create<ops::ReturnOp>(std::vector{ r });

    auto main_fun = mod.Create<ops::FuncOp>("main_fun",
                                            FunctionType::Get({ fieldType, fieldType, fieldType },
                                                              { fieldType }),
                                            true);
    auto rmain = main_fun.Create<ops::ApplyOp>(kernel_fun,
                                               std::vector{ main_fun.GetRegionArg(0),
                                                            main_fun.GetRegionArg(1) },
                                               std::vector{ main_fun.GetRegionArg(2) },
                                               std::vector<Value>{},
                                               std::vector<int64_t>{ 0 })
                     .GetResults()[0];
    main_fun.Create<ops::ReturnOp>(std::vector{ rmain });
    return mod;
}


void ThrowIfFailed(cudaError err) {
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}


TEST_CASE("CUDA", "[Program]") {
    auto mod = CreateModule();

    mlir::MLIRContext context;
    auto convertedModule = mlir::dyn_cast<mlir::ModuleOp>(ConvertOperation(context, mod));
    SnapshotIR(convertedModule);
    Compiler compiler{ TargetCUDAPipeline(context) };

    auto compiledModule = CompileWithStageResults(compiler, convertedModule);

    constexpr int size = 1000;
    std::vector<float> inputValues1(size);
    std::vector<float> inputValues2(size);
    std::vector<float> outputValues(size);
    std::vector<float> expectedValues(size);

    for (int i = 0; i < size; ++i) {
        inputValues1[i] = (i + 6) % 9;
        inputValues2[i] = (i + 9) % 7;
        expectedValues[i] = inputValues1[i] + inputValues2[i];
    }

    float* input1;
    float* input2;
    float* output;
    ThrowIfFailed(cudaMalloc(&input1, size * sizeof(float)));
    ThrowIfFailed(cudaMalloc(&input2, size * sizeof(float)));
    ThrowIfFailed(cudaMalloc(&output, size * sizeof(float)));
    ThrowIfFailed(cudaMemcpy(input1, inputValues1.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    ThrowIfFailed(cudaMemcpy(input2, inputValues2.data(), size * sizeof(float), cudaMemcpyHostToDevice));

    StridedMemRefType<float, 1> inputMemref1{ input1, input1, 0, { size }, { 1 } };
    StridedMemRefType<float, 1> inputMemref2{ input2, input2, 0, { size }, { 1 } };
    StridedMemRefType<float, 1> outputMemref{ output, output, 0, { size }, { 1 } };
    StridedMemRefType<float, 1> resultMemref{ nullptr, nullptr, 0, { 0 }, { 0 } };

    constexpr int optLevel = 3;
    sir::Runner jitRunner{ compiledModule, optLevel };
    jitRunner.Invoke("main_fun", inputMemref1, inputMemref2, outputMemref, Runner::Result{ resultMemref });

    ThrowIfFailed(cudaMemcpy(outputValues.data(), resultMemref.data, size * sizeof(float), cudaMemcpyDeviceToHost));
    ThrowIfFailed(cudaFree(input1));
    ThrowIfFailed(cudaFree(input2));
    ThrowIfFailed(cudaFree(output));

    const auto error = std::inner_product(
        outputValues.begin(),
        outputValues.end(),
        expectedValues.begin(),
        0.0f,
        [](auto acc, auto i) { return std::max(acc, i); },
        [](auto a, auto b) { return std::abs(a - b); });

    REQUIRE(error < 1e-5f);
}