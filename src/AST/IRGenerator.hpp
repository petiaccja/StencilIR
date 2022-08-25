#pragma once

#include <functional>
#include <stdexcept>
#include <tuple>
#include <typeindex>
#include <unordered_map>


template <class ASTNode, class IRResult, class Generator, class... ConcreteASTNodes>
class IRGenerator {
public:
    IRResult Generate(const ASTNode& node) const {
        auto generatorIt = m_generators.find(typeid(node));
        if (generatorIt == m_generators.end()) {
            throw std::invalid_argument("No generator registered for provided AST node type.");
        }
        return generatorIt->second(node);
    }

private:
    using GeneratorFunc = std::function<IRResult(const ASTNode&)>;
    const std::unordered_map<std::type_index, GeneratorFunc> m_generators = {
        { typeid(ConcreteASTNodes),
          [this](const ASTNode& node) {
              return static_cast<const Generator*>(this)->Generate(dynamic_cast<const ConcreteASTNodes&>(node));
          } }...
    };
};