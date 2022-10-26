#include "Checker.hpp"

#include <AST/ConvertASTToIR.hpp>

#include <llvm/FileCheck/FileCheck.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>


std::string PrintIr(mlir::ModuleOp module) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    module.print(ss);
    return s;
}


bool Check(std::string_view input, std::string_view pattern) {
    llvm::FileCheckRequest req;
    req.AllowEmptyInput = true;
    req.AllowUnusedPrefixes = true;
    req.EnableVarScope = false;
    req.AllowDeprecatedDagOverlap = false;
    req.Verbose = false;
    req.VerboseVerbose = false;
    req.NoCanonicalizeWhiteSpace = false;
    req.MatchFullLines = false;
    req.IgnoreCase = false;

    llvm::FileCheck FC(req);
    if (!FC.ValidateCheckPrefixes())
        return 2;

    llvm::Regex prefixRegex = FC.buildCheckPrefixRegex();
    std::string prefixRegexError;
    if (!prefixRegex.isValid(prefixRegexError)) {
        throw std::runtime_error("Prefix regex failed: " + prefixRegexError);
    }

    llvm::SourceMgr sourceManager;

    // Add the pattern to the source manager

    const auto patternStringRef = llvm::StringRef(pattern);
    auto patternBuffer = llvm::MemoryBuffer::getMemBuffer(patternStringRef);
    [[maybe_unused]] const unsigned patternBufferId = sourceManager.AddNewSourceBuffer(std::move(patternBuffer), llvm::SMLoc());

    std::pair<unsigned, unsigned> patternBufferIdRange;
    if (FC.readCheckFile(sourceManager, patternStringRef, prefixRegex, &patternBufferIdRange)) {
        throw std::runtime_error("Failed to read pattern file.");
    }

    // Add the input to the source manager
    const auto inputStringRef = llvm::StringRef(input);
    auto inputBuffer = llvm::MemoryBuffer::getMemBuffer(inputStringRef);
    sourceManager.AddNewSourceBuffer(std::move(inputBuffer), llvm::SMLoc());

    std::vector<llvm::FileCheckDiag> Diags;
    const bool success = FC.checkInput(sourceManager, inputStringRef, &Diags);
    return success;
}

bool Check(ast::Module& module, std::string_view pattern) {
    static mlir::MLIRContext context;
    auto ir = ConvertASTToIR(context, module);
    const auto str = PrintIr(ir);
    return Check(str, pattern);
}