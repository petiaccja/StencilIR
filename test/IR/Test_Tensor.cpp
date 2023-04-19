#include <TestTools/FileCheck.hpp>

#include <IR/Ops.hpp>

#include <catch2/catch.hpp>

using namespace sir;


TEST_CASE("Dim", "[DAG]") {
    auto mod = ops::ModuleOp();
    auto func = mod.Create<ops::FuncOp>("fn",
                                        FunctionType::Get(
                                            { FieldType::Get(Float32, 1),
                                              IndexType::Get() },
                                            { IndexType::Get() }));

    auto dim = func.Create<ops::DimOp>(func.GetRegionArg(0), func.GetRegionArg(1));
    func.Create<ops::ReturnOp>(std::vector{ dim.GetResults()[0] });

    const auto pattern = R"(
        // CHECK: func @fn(%[[TENSOR:.*]]: tensor<?xf32>, %[[INDEX:.*]]: index) -> index
        // CHECK-NEXT: %[[RES:.*]] = tensor.dim %[[TENSOR]], %[[INDEX]]
        // CHECK-NEXT: return %[[RES]]
    )";

    REQUIRE(CheckDAG(mod, pattern));
}


TEST_CASE("Alloc tensor", "[DAG]") {
    auto mod = ops::ModuleOp();
    auto func = mod.Create<ops::FuncOp>("fn",
                                        FunctionType::Get(
                                            { IndexType::Get() },
                                            { FieldType::Get(Float32, 1) }));

    auto alloc = func.Create<ops::AllocTensorOp>(Float32, std::vector{ func.GetRegionArg(0) });
    func.Create<ops::ReturnOp>(std::vector{ alloc.GetResults()[0] });

    const auto pattern = R"(
        // CHECK: func @fn(%[[SIZE:.*]]: index) -> tensor<?xf32>
        // CHECK-NEXT: %[[RES:.*]] = bufferization.alloc_tensor(%[[SIZE]]) : tensor<?xf32>
        // CHECK-NEXT: return %[[RES]]
    )";

    REQUIRE(CheckDAG(mod, pattern));
}


TEST_CASE("Extract slice", "[DAG]") {
    auto mod = ops::ModuleOp();
    auto func = mod.Create<ops::FuncOp>("fn",
                                        FunctionType::Get(
                                            { FieldType::Get(Float32, 1),
                                              IndexType::Get(),
                                              IndexType::Get(),
                                              IndexType::Get() },
                                            { FieldType::Get(Float32, 1) }));

    auto extract = func.Create<ops::ExtractSliceOp>(func.GetRegionArg(0),
                                                    std::vector{ func.GetRegionArg(1) },
                                                    std::vector{ func.GetRegionArg(2) },
                                                    std::vector{ func.GetRegionArg(3) });
    func.Create<ops::ReturnOp>(std::vector{ extract.GetResults()[0] });

    const auto pattern = R"(
        // CHECK: func @fn(%[[SOURCE:.*]]: tensor<?xf32>, %[[OFFSET:.*]]: index, %[[SIZE:.*]]: index, %[[STRIDE:.*]]: index) -> tensor<?xf32>
        // CHECK-NEXT: %[[RES:.*]] = tensor.extract_slice %[[SOURCE]][%[[OFFSET]]] [%[[SIZE]]] [%[[STRIDE]]]
        // CHECK-NEXT: return %[[RES]]
    )";

    REQUIRE(CheckDAG(mod, pattern));
}


TEST_CASE("Insert slice", "[DAG]") {
    auto mod = ops::ModuleOp();
    auto func = mod.Create<ops::FuncOp>("fn",
                                        FunctionType::Get(
                                            { FieldType::Get(Float32, 1),
                                              FieldType::Get(Float32, 1),
                                              IndexType::Get(),
                                              IndexType::Get(),
                                              IndexType::Get() },
                                            { FieldType::Get(Float32, 1) }));

    auto insert = func.Create<ops::InsertSliceOp>(func.GetRegionArg(0),
                                                  func.GetRegionArg(1),
                                                  std::vector{ func.GetRegionArg(2) },
                                                  std::vector{ func.GetRegionArg(3) },
                                                  std::vector{ func.GetRegionArg(4) });
    func.Create<ops::ReturnOp>(std::vector{ insert.GetResults()[0] });

    const auto pattern = R"(
        // CHECK: func @fn(%[[SOURCE:.*]]: tensor<?xf32>, %[[DEST:.*]]: tensor<?xf32>, %[[OFFSET:.*]]: index, %[[SIZE:.*]]: index, %[[STRIDE:.*]]: index) -> tensor<?xf32>
        // CHECK-NEXT: %[[RES:.*]] = tensor.insert_slice %[[SOURCE]] into %[[DEST]][%[[OFFSET]]] [%[[SIZE]]] [%[[STRIDE]]]
        // CHECK-NEXT: return %[[RES]]
    )";

    REQUIRE(CheckDAG(mod, pattern));
}