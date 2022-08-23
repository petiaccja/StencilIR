#pragma once

#include <functional>
#include <stdexcept>
#include <tuple>
#include <typeindex>
#include <unordered_map>


template <class ASTNode, class IROperation, class IRValue, class Generator, class... ConcreteASTNodes>
class IRGenerator {
public:
    std::tuple<IROperation, IRValue> Generate(const ASTNode& node) const {
        auto generatorIt = m_generators.find(typeid(node));
        if (generatorIt == m_generators.end()) {
            throw std::invalid_argument("No generator registered for provided AST node type.");
        }
        return generatorIt(node);
    }

private:
    using GeneratorFunc = std::function<std::tuple<IROperation, IRValue>(const ASTNode&)>;
    const std::unordered_map<std::type_index, GeneratorFunc> m_generators = {
        { typeid(ConcreteASTNodes),
          [this](const ConcreteASTNodes& node) {
              return static_cast<const Generator*>(this)->Generate(node);
          } }...
    };
};