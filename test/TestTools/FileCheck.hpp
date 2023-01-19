#pragma once

#include <AST/Nodes.hpp>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <filesystem>
#include <string_view>


struct Pass {
    explicit Pass(std::unique_ptr<mlir::Pass> pass) : pass(std::move(pass)) {}
    Pass(Pass&&) = default;
    virtual ~Pass() = default;
    virtual void MoveTo(mlir::PassManager& pm) {
        pm.addPass(std::move(pass));
    }

    std::unique_ptr<mlir::Pass> pass;
};

template <class OpT = void>
struct NestedPass : public Pass {
    using Pass::Pass;
    virtual void MoveTo(mlir::PassManager& pm) {
        pm.addNestedPass<OpT>(std::move(pass));
    }
};


bool CheckText(std::string_view input, std::string_view pattern);
bool CheckFile(std::filesystem::path file, std::vector<std::unique_ptr<Pass>> passes);
bool CheckAST(ast::Module& moduleNode, std::string_view pattern);

template <class... Passes>
auto CheckFile(std::filesystem::path file, Passes... passes) {
    std::vector<std::unique_ptr<Pass>> passVec;
    passVec.reserve(sizeof...(passes));
    (..., passVec.push_back(std::make_unique<Passes>(std::move(passes))));
    return CheckFile(file, std::move(passVec));
}