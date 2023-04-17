#pragma once


#include "Operation.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinDialect.h>

#include <typeindex>
#include <unordered_map>


namespace sir {


class Converter;


using ConverterFunction = std::function<mlir::Operation*(Converter&, Operation, mlir::ValueRange)>;


class Converter {
public:
    Converter(mlir::MLIRContext& context) : m_builder(&context) {}

    mlir::Operation* operator()(Operation operation);

    mlir::OpBuilder& Builder() { return m_builder; }
    void MapEntryBlock(const Region& region, mlir::Block& block);

    template <class ConcreteOp>
    void RegisterOp(ConverterFunction converterFunction) {
        m_converterFunctions.insert({ typeid(ConcreteOp), std::move(converterFunction) });
    }

private:
    std::unordered_map<std::type_index, ConverterFunction> m_converterFunctions;
    std::unordered_map<std::shared_ptr<ValueImpl>, mlir::Value> m_convertedResults;
    mlir::OpBuilder m_builder;
};


} // namespace sir