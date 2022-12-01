#pragma once

#include <AST/Nodes.hpp>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#include <filesystem>
#include <string_view>


bool CheckText(std::string_view input, std::string_view pattern);
bool CheckFile(std::filesystem::path file, std::vector<std::unique_ptr<mlir::Pass>> passes);
bool CheckAST(ast::Module& module, std::string_view pattern);

template <class... Passes>
auto CheckFile(std::filesystem::path file, Passes&&... passes) {
    std::vector<std::unique_ptr<mlir::Pass>> passVec;
    passVec.reserve(sizeof...(passes));
    (..., passVec.push_back(std::move(passes)));
    return CheckFile(file, std::move(passVec));
}