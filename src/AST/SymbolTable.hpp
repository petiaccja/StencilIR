#pragma once


#include <any>
#include <list>
#include <optional>
#include <stdexcept>
#include <unordered_map>


template <class Key, class Value>
class SymbolTable {
    struct Scope {
        std::unordered_map<Key, Value> symbols;
        std::any info;
    };

public:
    template <class Func>
    void RunInScope(Func func, std::any info = {}) {
        m_currentScope = m_scopes.emplace_back(Scope{ std::unordered_map<Key, Value>{}, info });
        func();
        m_currentScope = {};
        m_scopes.pop_back();
        if (!m_scopes.empty()) {
            m_currentScope = m_scopes.back();
        }
    }

    void Assign(Key symbolName, Value value) {
        if (!m_currentScope) {
            throw std::logic_error("You can only assign a value to a symbol from a RunInScope function.");
        }

        Scope& currentScope = m_currentScope.value();
        // Reassignment is okay.
        currentScope.symbols[symbolName] = value;
    }

    std::optional<std::reference_wrapper<Value>> Lookup(const Key& symbolName) {
        for (auto scopeIt = m_scopes.rbegin(); scopeIt != m_scopes.rend(); ++scopeIt) {
            auto symbolIt = scopeIt->symbols.find(symbolName);
            if (symbolIt != scopeIt->symbols.end()) {
                return symbolIt->second;
            }
        }
        return {};
    }

    std::any Info() const {
        if (!m_currentScope) {
            throw std::logic_error("No current scope to return info for.");
        }
        return m_currentScope.value().get().info;
    }

private:
    std::list<Scope> m_scopes;
    std::optional<std::reference_wrapper<Scope>> m_currentScope;
};