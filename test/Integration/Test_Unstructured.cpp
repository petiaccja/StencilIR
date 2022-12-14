#include "Utility/RunAST.hpp"

#include <AST/Building.hpp>
#include <AST/ConvertASTToIR.hpp>
#include <Compiler/Pipelines.hpp>
#include <Execution/Execution.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string_view>

#include <catch2/catch.hpp>


static std::shared_ptr<ast::Module> CreateAST() {
    auto cellK = ast::symref("cellK");
    auto edgeToCell = ast::symref("edgeToCell");
    auto cellWeights = ast::symref("cellWeights");

    // Stencil logic
    auto field = ast::symref("cellK");

    auto index = ast::index();

    auto connectivityIdx = ast::extend(ast::project(index, { 0 }), 1, ast::symref("elementIdx"));
    auto neighbourIdx = ast::sample(edgeToCell, connectivityIdx);
    auto neighbourWeight = ast::sample(cellWeights, connectivityIdx);
    auto cellIdx = ast::exchange(index, 0, neighbourIdx);
    auto neighbourValue = ast::sample(cellK, cellIdx);

    auto invalidIdx = ast::constant(ast::index_type, -1);
    auto isNeighbourValid = ast::neq(neighbourIdx, invalidIdx);
    auto accUpdated = ast::add(ast::symref("accumulator"),
                               ast::mul(
                                   neighbourWeight,
                                   neighbourValue));
    auto accSame = ast::symref("accumulator");
    auto acc = ast::if_(isNeighbourValid, { ast::yield({ accUpdated }) }, { ast::yield({ accSame }) });
    auto sum = ast::for_(ast::constant(ast::index_type, 0),
                         ast::dim(edgeToCell, ast::constant(ast::index_type, 1)),
                         ast::constant(ast::index_type, 1),
                         "elementIdx",
                         { ast::yield({ acc }) },
                         { ast::constant(0.0f) },
                         { "accumulator" });

    auto edge_diffs = ast::stencil("edge_diffs",
                                   {
                                       { "cellK", ast::FieldType{ ast::ScalarType::FLOAT32, 2 } },
                                       { "edgeToCell", ast::FieldType{ ast::ScalarType::INDEX, 2 } },
                                       { "cellWeights", ast::FieldType{ ast::ScalarType::FLOAT32, 2 } },
                                   },
                                   { ast::ScalarType::FLOAT32 },
                                   { ast::return_({ sum }) },
                                   2);

    // Main function logic
    auto outEdgeK = ast::symref("outEdgeK");

    auto apply = ast::apply(edge_diffs->name,
                            { cellK, edgeToCell, cellWeights },
                            { outEdgeK });

    auto main = ast::function("main",
                              {
                                  { "cellK", ast::FieldType{ ast::ScalarType::FLOAT32, 2 } },
                                  { "edgeToCell", ast::FieldType{ ast::ScalarType::INDEX, 2 } },
                                  { "cellWeights", ast::FieldType{ ast::ScalarType::FLOAT32, 2 } },
                                  { "outEdgeK", ast::FieldType{ ast::ScalarType::FLOAT32, 2 } },
                              },
                              {},
                              { apply, ast::return_() });

    return ast::module_({ main },
                        { edge_diffs });
}


TEST_CASE("Unstructured", "[Program]") {
    constexpr ptrdiff_t numEdges = 5;
    constexpr ptrdiff_t numCells = numEdges + 1;
    constexpr ptrdiff_t numLevels = 3;
    std::array<float, numCells * numLevels> cellK;
    std::array<float, numEdges * numLevels> edgeK;
    std::array<ptrdiff_t, numEdges * 2> edgeToCell;
    std::array<float, numEdges * 2> cellWeights;

    std::iota(cellK.begin(), cellK.end(), 0.0f);
    std::for_each(cellK.begin(), cellK.end(), [](float& v) { v = std::pow(v, 1.1f); });
    std::fill(edgeK.begin(), edgeK.end(), -1.0f);
    std::iota(edgeToCell.begin(), edgeToCell.begin() + numEdges, 0);
    std::iota(edgeToCell.begin() + numEdges, edgeToCell.end(), 1);
    std::fill(cellWeights.begin(), cellWeights.begin() + numEdges, -1.0f);
    std::fill(cellWeights.begin() + numEdges, cellWeights.end(), 1.0f);


    MemRef<float, 2> cellKMem{ cellK.data(), cellK.data(), 0, { numCells, numLevels }, { 1, numCells } };
    MemRef<float, 2> edgeKMem{ edgeK.data(), edgeK.data(), 0, { numEdges, numLevels }, { 1, numEdges } };
    MemRef<ptrdiff_t, 2> edgeToCellMem{ edgeToCell.data(), edgeToCell.data(), 0, { numEdges, 2 }, { 1, numEdges } };
    MemRef<float, 2> cellWeightsMem{ cellWeights.data(), cellWeights.data(), 0, { numEdges, 2 }, { 1, numEdges } };

    const auto program = CreateAST();
    const auto stages = RunAST(*program, "main", cellKMem, edgeToCellMem, cellWeightsMem, edgeKMem);

    const std::array<float, numEdges* numLevels> expectedBuffer = {
        1.f, 1.14354706f, 1.20482254f, 1.2464242f, 1.27830124f,
        1.32631063f, 1.34545708f, 1.3624239f, 1.37767506f, 1.39154434f,
        1.41603279f, 1.42697716f, 1.43721581f, 1.44683456f, 1.45591354f
    };

    const auto maxDifference = std::inner_product(
        edgeK.begin(),
        edgeK.end(),
        expectedBuffer.begin(),
        0.0f,
        [](float acc, float v) { return std::max(acc, v); },
        [](float u, float v) { return std::abs(u - v); });

    std::stringstream ss;
    for (auto& stage : stages) {
        ss << "// " << stage.name << std::endl;
        ss << stage.ir << "\n"
           << std::endl;
    }
    INFO(ss.str());
    REQUIRE(maxDifference < 0.001f);
}
