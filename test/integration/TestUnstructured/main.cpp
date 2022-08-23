#include <AST/AST.hpp>
#include <AST/Build.hpp>
#include <AST/LowerToIR.hpp>
#include <AST/Node.hpp>
#include <AST/Types.hpp>
#include <Compiler/Lowering.hpp>
#include <Execution/Execution.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string_view>


std::shared_ptr<ast::Module> CreateLaplacian() {
    auto cellK = ast::symref("cellK");
    auto edgeToCell = ast::symref("edgeToCell");
    auto cellWeights = ast::symref("cellWeights");

    // Kernel logic
    auto field = ast::symref("cellK");

    auto c0 = ast::constant(0, types::FundamentalType::SSIZE);
    auto c1 = ast::constant(1, types::FundamentalType::SSIZE);

    auto weightLeft = ast::sample_indirect(ast::index(), 0, cellWeights, c0);
    auto weightRight = ast::sample_indirect(ast::index(), 0, cellWeights, c1);
    auto sampleLeft = ast::sample(cellK, ast::jump_indirect(ast::index(), 0, edgeToCell, c0));
    auto sampleRight = ast::sample(cellK, ast::jump_indirect(ast::index(), 0, edgeToCell, c1));
    auto sum = ast::sub(ast::mul(sampleLeft, weightLeft),
                        ast::mul(sampleRight, weightRight));

    auto ret = ast::return_({ sum });

    auto kernel = ast::stencil("edge_diffs",
                               {
                                   { "cellK", types::FieldType{ types::FundamentalType::FLOAT32, 2 } },
                                   { "edgeToCell", types::FieldType{ types::FundamentalType::SSIZE, 2 } },
                                   { "cellWeights", types::FieldType{ types::FundamentalType::FLOAT32, 2 } },
                               },
                               { types::FundamentalType::FLOAT32 },
                               { ret },
                               2);

    // Main function logic
    auto outEdgeK = ast::symref("outEdgeK");

    auto applyStencil = ast::apply(kernel->name,
                                   { cellK, edgeToCell, cellWeights },
                                   { outEdgeK });

    return ast::module_({ applyStencil },
                        { kernel },
                        {
                            { "cellK", types::FieldType{ types::FundamentalType::FLOAT32, 2 } },
                            { "edgeToCell", types::FieldType{ types::FundamentalType::SSIZE, 2 } },
                            { "cellWeights", types::FieldType{ types::FundamentalType::FLOAT32, 2 } },
                            { "outEdgeK", types::FieldType{ types::FundamentalType::FLOAT32, 2 } },
                        });
}


void RunLaplacian(JitRunner& runner) {
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
    std::fill(cellWeights.begin() + numEdges, cellWeights.end(), -1.0f);


    MemRef<float, 2> cellKMem{ cellK.data(), cellK.data(), 0, { numCells, numLevels }, { 1, numCells } };
    MemRef<float, 2> edgeKMem{ edgeK.data(), edgeK.data(), 0, { numEdges, numLevels }, { 1, numEdges } };
    MemRef<ptrdiff_t, 2> edgeToCellMem{ edgeToCell.data(), edgeToCell.data(), 0, { numEdges, 2 }, { 1, numEdges } };
    MemRef<float, 2> cellWeightsMem{ cellWeights.data(), cellWeights.data(), 0, { numEdges, 2 }, { 1, numEdges } };

    runner.InvokeFunction("main", cellKMem, edgeToCellMem, cellWeightsMem, edgeKMem);

    std::cout << "Input:" << std::endl;
    for (size_t level = 0; level < numLevels; ++level) {
        for (size_t cell = 0; cell < numCells; ++cell) {
            std::cout << std::setprecision(4) << cellK[level * numCells + cell] << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << "\nOutput:" << std::endl;
    for (size_t level = 0; level < numLevels; ++level) {
        for (size_t edge = 0; edge < numEdges; ++edge) {
            std::cout << std::setprecision(4) << edgeK[level * numEdges + edge] << "\t";
        }
        std::cout << std::endl;
    }
}

std::string StringizeIR(mlir::ModuleOp ir) {
    std::string s;
    llvm::raw_string_ostream ss{ s };
    ir->print(ss, mlir::OpPrintingFlags{}.elideLargeElementsAttrs());
    return s;
}

void DumpIR(std::string_view ir, std::string_view name) {
    assert(!name.empty());
    std::string fname;
    std::transform(name.begin(), name.end(), std::back_inserter(fname), [](char c) {
        if (std::ispunct(c) || std::isspace(c)) {
            c = '_';
        }
        return std::tolower(c);
    });
    auto tempfile = std::filesystem::temp_directory_path() / (fname + ".mlir");
    std::ofstream of(tempfile, std::ios::trunc);
    of << ir;
}


int main() {
    mlir::MLIRContext context;

    std::shared_ptr<ast::Module> ast = CreateLaplacian();
    try {
        mlir::ModuleOp module = LowerToIR(context, *ast);
        ApplyLocationSnapshot(context, module);
        DumpIR(StringizeIR(module), "Stencil original");
        ApplyCleanupPasses(context, module);
        DumpIR(StringizeIR(module), "Stencil cleaned");

        auto llvmCpuStages = LowerToLLVMCPU(context, module);
        for (auto& stage : llvmCpuStages) {
            DumpIR(StringizeIR(stage.second), stage.first);
        }

        constexpr int optLevel = 3;
        JitRunner jitRunner{ llvmCpuStages.back().second, optLevel };

        DumpIR(jitRunner.LLVMIR(), "LLVM IR");

        RunLaplacian(jitRunner);
    }
    catch (std::exception& ex) {
        std::cout << ex.what() << std::endl;
    }
}