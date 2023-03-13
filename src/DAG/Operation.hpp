#pragma once


#include <AST/Types.hpp>

#include <algorithm>
#include <any>
#include <memory>
#include <optional>
#include <span>
#include <typeindex>
#include <unordered_set>
#include <vector>


namespace dag {


struct Location {
    std::string file;
    int line;
    int col;
};


struct ValueImpl;
struct OperationImpl;

class Value;
class Operation;


struct Region {
    std::vector<Value> args;
    std::vector<Operation> operations;
};


struct ValueImpl {
    std::weak_ptr<OperationImpl> def;
    std::unordered_set<std::shared_ptr<OperationImpl>> users;
    size_t index;
};


class Value {
public:
    Value(Operation def, size_t index);
    Value(std::shared_ptr<ValueImpl> impl) : impl(impl) {}
    operator std::shared_ptr<ValueImpl>() const { return impl; }

    Operation DefiningOp() const;
    size_t Index() const;
    void AddUser(Operation user);
    void RemoveUser(Operation user);

private:
    std::shared_ptr<ValueImpl> impl;
};


class Operand {
public:
    Operand(Value source, Operation owner);
    ~Operand();
    Value source;
    std::weak_ptr<OperationImpl> owner;
};


struct OperationImpl {
    std::type_index m_type;
    std::vector<Operand> m_operands;
    std::vector<Value> m_results;
    std::vector<Region> m_regions;
    std::any m_attributes;
    std::optional<Location> m_loc;
};


class Operation {
public:
    template <class ConcreteOp>
    Operation(ConcreteOp& op,
              std::vector<Value> operands,
              size_t numResults,
              std::vector<Region> regions,
              std::any attributes,
              std::optional<Location> loc = {})
        : impl(std::make_shared<OperationImpl>(OperationImpl{ typeid(op),
                                                              {},
                                                              {},
                                                              std::move(regions),
                                                              attributes,
                                                              loc })) {
        for (auto& value : operands) {
            impl->m_operands.push_back(Operand(value, *this));
        }
        for (size_t i = 0; i < numResults; ++i) {
            impl->m_results.push_back(Value(*this, i));
        }
    }


    Operation(std::shared_ptr<OperationImpl> impl) : impl(impl) {}
    operator std::shared_ptr<OperationImpl>() const { return impl; }

    std::type_index Type() const;
    std::span<Operand> Operands() const;
    std::span<Value> Results() const;
    std::span<Region> Regions() const;
    const std::any& Attributes() const;
    const std::optional<Location>& Location() const;

private:
    std::shared_ptr<OperationImpl> impl;
};


} // namespace dag