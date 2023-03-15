#include <DAG/Ops.hpp>
#include <TestTools/FileCheck.hpp>

#include <catch2/catch.hpp>


TEST_CASE("Dim", "[DAG]") {
    auto mod = dag::ModuleOp();
    auto func = mod.Create<dag::FuncOp>("fn",
                                        ast::FunctionType::Get(
                                            { ast::FieldType::Get(ast::Float32, 1),
                                              ast::IndexType::Get() },
                                            { ast::IndexType::Get() }));

    auto dim = func.Create<dag::DimOp>(func.GetRegionArg(0), func.GetRegionArg(1));
    func.Create<dag::ReturnOp>(std::vector{ dim.GetResults()[0] });

    const auto pattern = R"(
        // CHECK: func @fn(%[[TENSOR:.*]]: tensor<?xf32>, %[[INDEX:.*]]: index) -> index
        // CHECK-NEXT: %[[RES:.*]] = tensor.dim %[[TENSOR]], %[[INDEX]]
        // CHECK-NEXT: return %[[RES]]
    )";

    REQUIRE(CheckDAG(mod, pattern));
}


TEST_CASE("Alloc tensor", "[DAG]") {
    auto mod = dag::ModuleOp();
    auto func = mod.Create<dag::FuncOp>("fn",
                                        ast::FunctionType::Get(
                                            { ast::IndexType::Get() },
                                            { ast::FieldType::Get(ast::Float32, 1) }));

    auto alloc = func.Create<dag::AllocTensorOp>(ast::Float32, std::vector{ func.GetRegionArg(0) });
    func.Create<dag::ReturnOp>(std::vector{ alloc.GetResults()[0] });

    const auto pattern = R"(
        // CHECK: func @fn(%[[SIZE:.*]]: index) -> tensor<?xf32>
        // CHECK-NEXT: %[[RES:.*]] = bufferization.alloc_tensor(%[[SIZE]]) : tensor<?xf32>
        // CHECK-NEXT: return %[[RES]]
    )";

    REQUIRE(CheckDAG(mod, pattern));
}


TEST_CASE("Extract slice", "[DAG]") {
    auto mod = dag::ModuleOp();
    auto func = mod.Create<dag::FuncOp>("fn",
                                        ast::FunctionType::Get(
                                            { ast::FieldType::Get(ast::Float32, 1),
                                              ast::IndexType::Get(),
                                              ast::IndexType::Get(),
                                              ast::IndexType::Get() },
                                            { ast::FieldType::Get(ast::Float32, 1) }));

    auto extract = func.Create<dag::ExtractSliceOp>(func.GetRegionArg(0),
                                                    std::vector{ func.GetRegionArg(1) },
                                                    std::vector{ func.GetRegionArg(2) },
                                                    std::vector{ func.GetRegionArg(3) });
    func.Create<dag::ReturnOp>(std::vector{ extract.GetResults()[0] });

    const auto pattern = R"(
        // CHECK: func @fn(%[[SOURCE:.*]]: tensor<?xf32>, %[[OFFSET:.*]]: index, %[[SIZE:.*]]: index, %[[STRIDE:.*]]: index) -> tensor<?xf32>
        // CHECK-NEXT: %[[RES:.*]] = tensor.extract_slice %[[SOURCE]][%[[OFFSET]]] [%[[SIZE]]] [%[[STRIDE]]]
        // CHECK-NEXT: return %[[RES]]
    )";

    REQUIRE(CheckDAG(mod, pattern));
}


TEST_CASE("Insert slice", "[DAG]") {
    auto mod = dag::ModuleOp();
    auto func = mod.Create<dag::FuncOp>("fn",
                                        ast::FunctionType::Get(
                                            { ast::FieldType::Get(ast::Float32, 1),
                                              ast::FieldType::Get(ast::Float32, 1),
                                              ast::IndexType::Get(),
                                              ast::IndexType::Get(),
                                              ast::IndexType::Get() },
                                            { ast::FieldType::Get(ast::Float32, 1) }));

    auto insert = func.Create<dag::InsertSliceOp>(func.GetRegionArg(0),
                                                  func.GetRegionArg(1),
                                                  std::vector{ func.GetRegionArg(2) },
                                                  std::vector{ func.GetRegionArg(3) },
                                                  std::vector{ func.GetRegionArg(4) });
    func.Create<dag::ReturnOp>(std::vector{ insert.GetResults()[0] });

    const auto pattern = R"(
        // CHECK: func @fn(%[[SOURCE:.*]]: tensor<?xf32>, %[[DEST:.*]]: tensor<?xf32>, %[[OFFSET:.*]]: index, %[[SIZE:.*]]: index, %[[STRIDE:.*]]: index) -> tensor<?xf32>
        // CHECK-NEXT: %[[RES:.*]] = tensor.insert_slice %[[SOURCE]] into %[[DEST]][%[[OFFSET]]] [%[[SIZE]]] [%[[STRIDE]]]
        // CHECK-NEXT: return %[[RES]]
    )";

    REQUIRE(CheckDAG(mod, pattern));
}