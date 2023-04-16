#include "FileCheck.hpp"

#include <AST/ConvertASTToIR.hpp>
#include <IR/ConvertOps.hpp>
#include <IR/Operation.hpp>
#include <Diagnostics/Exception.hpp>
#include <Diagnostics/Formatting.hpp>
#include <Diagnostics/Handlers.hpp>
#include <Dialect/Stencil/IR/StencilOps.hpp>
#include <Dialect/Stencil/Transforms/BufferizableOpInterfaceImpl.hpp>

#include <llvm/FileCheck/FileCheck.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>

#include <fstream>


static std::string PrintOp(mlir::Operation* op) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    op->print(ss);
    return s;
}


std::string_view FormatMatchType(llvm::FileCheckDiag::MatchType matchType) {
    switch (matchType) {
        case llvm::FileCheckDiag::MatchFoundAndExpected: return "expected pattern found";
        case llvm::FileCheckDiag::MatchFoundButExcluded: return "excluded pattern found";
        case llvm::FileCheckDiag::MatchFoundButWrongLine: return "expected pattern on the wrong line";
        case llvm::FileCheckDiag::MatchFoundButDiscarded: return "expected pattern discarded";
        case llvm::FileCheckDiag::MatchFoundErrorNote: return "pattern found but an error occured";
        case llvm::FileCheckDiag::MatchNoneAndExcluded: return "excluded pattern not found";
        case llvm::FileCheckDiag::MatchNoneButExpected: return "expected pattern not found";
        case llvm::FileCheckDiag::MatchNoneForInvalidPattern: return "invalid pattern";
        case llvm::FileCheckDiag::MatchFuzzy: return "fuzzy match";
    }
    return "unknown error";
}


bool CheckText(std::string_view input, std::string_view pattern) {
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

    std::vector<llvm::FileCheckDiag> diags;
    const bool success = FC.checkInput(sourceManager, inputStringRef, &diags);
    if (!success) {
        std::stringstream message;
        for (auto& diag : diags) {
            std::optional<std::string> location;
            if (const char* locPtr = diag.CheckLoc.getPointer()) {
                assert(locPtr - pattern.data() < std::ssize(pattern));
                const auto line = std::count(pattern.data(), locPtr, '\n') + 1;
                const auto lineStart = std::find(std::reverse_iterator{ locPtr }, std::reverse_iterator{ pattern.data() }, '\n');
                const auto column = locPtr - lineStart.base();
                location = FormatLocation("-", line, column);
            }
            const auto note = FormatMatchType(diag.MatchTy).data() + std::string(diag.Note.empty() ? "" : ": ") + diag.Note;
            message << FormatDiagnostic(
                location,
                {},
                note)
                    << std::endl;
        }
        message << std::endl
                << input << std::endl;
        throw Exception(message.str());
    }
    return success;
}


bool CheckAST(ast::Module& moduleNode, std::string_view pattern) {
    static mlir::MLIRContext context;
    auto ir = ConvertASTToIR(context, moduleNode);
    const auto str = PrintOp(ir);
    return CheckText(str, pattern);
}


bool CheckDAG(dag::Operation moduleNode, std::string_view pattern) {
    static mlir::MLIRContext context;
    auto ir = dag::ConvertOperation(context, moduleNode);
    const auto str = PrintOp(ir);
    return CheckText(str, pattern);
}


bool CheckFile(const std::filesystem::path& file, std::vector<std::unique_ptr<Pass>>&& passes) {
    std::ifstream is(file);
    if (!is.is_open()) {
        throw std::runtime_error("failed to open file: " + file.string());
    }
    is.seekg(0, std::ios::end);
    const size_t length = is.tellg();
    is.seekg(0, std::ios::beg);
    std::string source(length, ' ');
    is.read(source.data(), length);

    // Load all dialects
    mlir::MLIRContext context;
    mlir::registerAllDialects(context);
    context.getOrLoadDialect<stencil::StencilDialect>();
    mlir::DialectRegistry registry;
    mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
    stencil::registerBufferizableOpInterfaceExternalModels(registry);
    context.appendDialectRegistry(registry);

    mlir::ParserConfig config{ &context };

    mlir::OwningOpRef<mlir::ModuleOp> ref = nullptr;
    {
        ScopedDiagnosticCollector diagnostics{ context };
        ref = mlir::parseSourceString<mlir::ModuleOp>(source, config);
        if (!ref) {
            auto diagList = diagnostics.TakeDiagnostics();
            throw CompilationError(diagList);
        }
    }
    mlir::ModuleOp op = ref.get();

    mlir::PassManager pm{ &context };
    for (auto& pass : passes) {
        pass->MoveTo(pm);
    }

    {
        ScopedDiagnosticCollector diagnostics{ context };
        auto result = pm.run(op);
        if (failed(result)) {
            auto diagList = diagnostics.TakeDiagnostics();
            throw CompilationError(diagList, op);
        }
    }

    const auto moduleStr = PrintOp(op);
    return CheckText(moduleStr, source);
}