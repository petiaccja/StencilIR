#include "Converter.hpp"

#include <mlir/IR/Operation.h>

#include <algorithm>
#include <ranges>


namespace dag {


mlir::Operation* Converter::operator()(Operation operation) {
    const auto type = operation.Type();
    const auto converterFunIt = m_converterFunctions.find(type);
    if (converterFunIt == m_converterFunctions.end()) {
        throw std::invalid_argument("no converter function for given operation");
    }
    const auto& converterFun = converterFunIt->second;

    mlir::SmallVector<mlir::Value> convertedOperands;
    const auto operands = operation.Operands();
    convertedOperands.reserve(operands.size());

    std::ranges::transform(operands, std::back_inserter(convertedOperands), [this](const auto& operand) {
        auto valueIt = m_convertedResults.find(operand);
        if (valueIt != m_convertedResults.end()) {
            throw std::invalid_argument("operations are not topologically sorted");
        }
        return valueIt->second;
    });

    const auto convertedOp = converterFun(*this, operation, convertedOperands);
    m_builder.setInsertionPointAfter(convertedOp);
    if (convertedOp->getNumResults() != operation.Results().size()) {
        throw std::invalid_argument("invalid conversion: result count mismatch");
    }

    auto resultIt = operation.Results().begin();
    auto convertedResultIt = convertedOp->getResults().begin();
    for (; resultIt != operation.Results().end(); ++resultIt, ++convertedResultIt) {
        m_convertedResults.insert({ *resultIt, *convertedResultIt });
    }
    return convertedOp;
}


void Converter::MapEntryBlock(const Region& region, mlir::Block& block) {
    if (block.getNumArguments() != region.args.size()) {
        throw std::invalid_argument("region and block must have the same number of arguments");
    }
    auto regionArgIt = region.args.begin();
    auto blockArgIt = block.args_begin();
    for (; regionArgIt != region.args.end(); ++regionArgIt, ++blockArgIt) {
        m_convertedResults.insert({ *regionArgIt, *blockArgIt });
    }
}


} // namespace dag