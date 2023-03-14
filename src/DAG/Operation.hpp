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

    template <class OpT, class... Args>
    OpT Create(Args&&... args) {
        auto op = OpT(std::forward<Args>(args)...);
        operations.push_back(op);
        return op;
    }
};


struct ValueImpl {
    std::weak_ptr<OperationImpl> def;
    std::unordered_set<std::shared_ptr<OperationImpl>> users;
    size_t index;
};


class Value {
public:
    Value(Operation owner, size_t index);
    Value(std::shared_ptr<ValueImpl> impl) : impl(impl) {}
    operator std::shared_ptr<ValueImpl>() const { return impl; }

    Operation GetOwner() const;
    size_t GetIndex() const;
    void AddUser(Operation user);
    void RemoveUser(Operation user);

private:
    std::shared_ptr<ValueImpl> impl;
};


class Operand {
public:
    Operand(Value source, Operation owner);
    ~Operand();

    Value GetSource() const { return source; }
    std::weak_ptr<OperationImpl> GetOwner() const { return owner; }

private:
    Value source;
    std::weak_ptr<OperationImpl> owner;
};


struct OperationImpl {
    std::type_index type;
    std::vector<Operand> operands;
    std::vector<Value> results;
    std::vector<Region> regions;
    std::any attributes;
    std::optional<Location> loc;
};


class Operation {
public:
    Operation(std::type_index type,
              std::vector<Value> operands,
              size_t numResults,
              std::vector<Region> regions,
              std::any attributes,
              std::optional<Location> loc = {});


    Operation(std::shared_ptr<OperationImpl> impl) : impl(impl) {}
    operator std::shared_ptr<OperationImpl>() const { return impl; }

    std::type_index Type() const;
    std::span<Operand> GetOperands() const;
    std::span<Value> GetResults() const;
    std::span<Region> GetRegions() const;
    const std::any& GetAttributes() const;
    const std::optional<Location>& GetLocation() const;

private:
    std::shared_ptr<OperationImpl> impl;
};


} // namespace dag