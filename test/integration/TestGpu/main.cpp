#include <AST/ASTBuilding.hpp>
#include <AST/ConvertASTToIR.hpp>
#include <Execution/Execution.hpp>

#include <iostream>
#include <string_view>


std::shared_ptr<ast::Module> CreateDdx() {
    // Kernel logic
    const auto field = ast::symref("field");
    const auto index = ast::index();
    const auto left = ast::jump(index, { -1, 0 });
    const auto right = ast::jump(index, { 1, 0 });
    const auto sleft = ast::sample(field, left);
    const auto sright = ast::sample(field, right);
    const auto ddx = ast::sub(sright, sleft);
    auto ret = ast::return_({ ddx });
    auto kernel = ast::stencil("ddx",
                               { { "field", types::FieldType{ types::FundamentalType::FLOAT32, 2 } } },
                               { types::FundamentalType::FLOAT32 },
                               { ret },
                               2);

    // Main function logic
    auto inputField = ast::symref("input");
    auto output = ast::symref("out");

    auto kernelLaunch = ast::apply(kernel->name,
                                   { inputField },
                                   { output });

    return ast::module_({ kernelLaunch },
                        { kernel },
                        {
                            { "input", types::FieldType{ types::FundamentalType::FLOAT32, 2 } },
                            { "out", types::FieldType{ types::FundamentalType::FLOAT32, 2 } },
                        });
}

void DumpIR(mlir::ModuleOp ir, std::string_view name = {}) {
    if (!name.empty()) {
        std::cout << name << ":\n"
                  << std::endl;
    }
    ir->dump();
    std::cout << "\n"
              << std::endl;
}

int main() {
    mlir::MLIRContext context;

    std::shared_ptr<ast::Module> ast = CreateDdx();
    try {

    }
    catch (std::exception& ex) {
        std::cout << ex.what() << std::endl;
    }
}