#include "Converter.hpp"

#include <Diagnostics/Exception.hpp>
#include <Diagnostics/Formatting.hpp>

#include <mlir/IR/Operation.h>

#include <algorithm>
#include <ranges>


namespace dag {



static mlir::Location ConvertLocation(mlir::OpBuilder& builder, const std::optional<Location>& location) {
    if (location) {
        auto fileattr = builder.getStringAttr(location->file);
        return mlir::FileLineColLoc::get(fileattr, location->line, location->col);
    }
    return builder.getUnknownLoc();
}



mlir::Operation* Converter::operator()(Operation operation) {
    const auto type = operation.Type();
    const auto converterFunIt = m_converterFunctions.find(type);
    if (converterFunIt == m_converterFunctions.end()) {
        mlir::Diagnostic diag(ConvertLocation(m_builder, operation.GetLocation()), mlir::DiagnosticSeverity::Error);
        diag << "no converter for given operation";
        diag.attachNote() << "operation's RTTI type is " << operation.Type().name();
        std::vector<mlir::Diagnostic> diags;
        diags.push_back(std::move(diag));
        throw CompilationError(diags);
    }
    const auto& converterFun = converterFunIt->second;

    mlir::SmallVector<mlir::Value> convertedOperands;
    const auto& operands = operation.GetOperands();
    convertedOperands.reserve(operands.size());

    std::ranges::transform(operands, std::back_inserter(convertedOperands), [this, &operation](const auto& operand) {
        auto valueIt = m_convertedResults.find(operand.GetSource());
        if (valueIt == m_convertedResults.end()) {
            mlir::Diagnostic diag(ConvertLocation(m_builder, operation.GetLocation()), mlir::DiagnosticSeverity::Error);
            diag << "operation not preceded by its arguments";
            diag.attachNote() << "current operation's RTTI type is " << operation.Type().name();
            std::vector<mlir::Diagnostic> diags;
            diags.push_back(std::move(diag));
            throw CompilationError(diags);
        }
        return valueIt->second;
    });

    const auto convertedOp = converterFun(*this, operation, convertedOperands);
    m_builder.setInsertionPointAfter(convertedOp);
    if (convertedOp->getNumResults() != operation.GetResults().size()) {
        throw std::invalid_argument("invalid conversion: result count mismatch");
    }

    auto resultIt = operation.GetResults().begin();
    auto convertedResultIt = convertedOp->getResults().begin();
    for (; resultIt != operation.GetResults().end(); ++resultIt, ++convertedResultIt) {
        m_convertedResults.insert({ *resultIt, *convertedResultIt });
    }
    return convertedOp;
}


void Converter::MapEntryBlock(const Region& region, mlir::Block& block) {
    if (block.getNumArguments() != region.GetArgs().size()) {
        throw std::invalid_argument("region and block must have the same number of arguments");
    }
    auto regionArgIt = region.GetArgs().begin();
    auto blockArgIt = block.args_begin();
    for (; regionArgIt != region.GetArgs().end(); ++regionArgIt, ++blockArgIt) {
        m_convertedResults.insert({ *regionArgIt, *blockArgIt });
    }
}


} // namespace dag