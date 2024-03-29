#pragma once

#include <IR/Operation.hpp>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>

#include <filesystem>
#include <string_view>


using namespace sir;


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
bool CheckFile(mlir::MLIRContext& context, const std::filesystem::path& file, std::vector<std::unique_ptr<Pass>>&& passes);
bool CheckDAG(Operation moduleNode, std::string_view pattern);
template <class... Passes>
auto CheckFile(mlir::MLIRContext& context, const std::filesystem::path& file, Passes... passes) {
    std::vector<std::unique_ptr<Pass>> passVec;
    passVec.reserve(sizeof...(passes));
    (..., passVec.push_back(std::make_unique<Passes>(std::move(passes))));
    return CheckFile(context, file, std::move(passVec));
}
inline bool CheckFile(const std::filesystem::path& file, std::vector<std::unique_ptr<Pass>>&& passes) {
    mlir::MLIRContext context;
    return CheckFile(context, file, std::move(passes));
}
template <class... Passes>
auto CheckFile(const std::filesystem::path& file, Passes... passes) {
    mlir::MLIRContext context;
    return CheckFile(context, file, std::move(passes)...);
}