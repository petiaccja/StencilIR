#include "Utility.hpp"

#include <regex>


mlir::StringAttr UniqueStencilName(stencil::StencilOp originalStencil, mlir::PatternRewriter& rewriter) {
    using namespace std::string_literals;

    constexpr std::string_view suffix = "proc";

    std::string symName = originalStencil.getSymName().str();
    std::regex format{ R"((.*)()_)"s + suffix.data() + R"(_([0-9]+))" };
    std::smatch match;
    std::regex_match(symName, match, format);
    if (!match.empty()) {
        symName = match[1];
    }

    size_t serial = 1;
    mlir::StringAttr fusedSymName;
    do {
        const auto newName = symName + "_" + suffix.data() + "_" + std::to_string(serial);
        fusedSymName = rewriter.getStringAttr(newName);
        ++serial;
    } while (nullptr != mlir::SymbolTable::lookupNearestSymbolFrom(originalStencil, fusedSymName));
    return fusedSymName;
}