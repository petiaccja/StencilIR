#pragma once


#include <list>
#include <optional>
#include <stdexcept>
#include <unordered_map>


template <class Key, class Value>
class SymbolTable {
    using MapT = std::unordered_map<Key, Value>;

public:
    template <class Func>
    void RunInScope(Func func) {
        m_currentScope = m_scopes.emplace_back();
        func();
        m_currentScope = nullptr;
        m_scopes.pop_back();
        if (!m_scopes.empty()) {
            m_currentScope = m_scopes.back();
        }
    }

    void Assign(Key symbolName, Value value) {
        if (!m_currentScope) {
            throw std::logic_error("You can only assign a value to a symbol from a RunInScope function.");
        }

        MapT& currentScope = m_currentScope.value();
        // Reassignment is okay.
        currentScope.find[symbolName] = value;
    }

    std::optional<std::reference_wrapper<Value>> Lookup(const Key& symbolName) {
        for (auto scopeIt = m_scopes.rbegin(); scopeIt != m_scopes.rend(); ++scopeIt) {
            auto symbolIt = scopeIt->find(symbolName);
            if (symbolIt != scopeIt->end()) {
                return symbolIt->second;
            }
        }
        return {};
    }

private:
    std::list<MapT> m_scopes;
    std::optional<std::reference_wrapper<Value>> m_currentScope;
};