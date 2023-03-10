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


class OperandImpl;
class ResultImpl;
class OperationImpl;


struct RegionImpl {
    std::vector<std::shared_ptr<ResultImpl>> args;
    std::vector<std::shared_ptr<OperationImpl>> operations;
};


class ResultImpl : std::enable_shared_from_this<OperandImpl> {
public:
    ResultImpl(std::weak_ptr<OperationImpl> def, size_t index)
        : m_def(def), m_index(index) {}

    std::shared_ptr<OperationImpl> DefiningOp() const {
        auto locked = m_def.lock();
        if (!locked) {
            throw std::logic_error("operation that created this value has been deleted");
        }
        return locked;
    }

    size_t Index() const {
        return m_index;
    }

    void AddUser(std::shared_ptr<OperandImpl> user) {
        m_users.insert(user);
    }

    void RemoveUser(std::shared_ptr<OperandImpl> user) {
        m_users.erase(user);
    }

private:
    std::weak_ptr<OperationImpl> m_def;
    std::unordered_set<std::shared_ptr<OperandImpl>> m_users;
    size_t m_index;
};


class OperandImpl : std::enable_shared_from_this<OperandImpl> {
public:
    OperandImpl(std::shared_ptr<ResultImpl> result, std::weak_ptr<OperationImpl> user)
        : m_result(result), m_user(user) {
        m_result->AddUser(shared_from_this());
    }

    ~OperandImpl() {
        m_result->RemoveUser(shared_from_this());
    }

    auto Result() { return m_result; }
    auto User() { return m_user; }

private:
    std::shared_ptr<ResultImpl> m_result;
    std::weak_ptr<OperationImpl> m_user;
};



class OperationImpl : std::enable_shared_from_this<OperationImpl> {
public:
    template <class ConcreteOp>
    OperationImpl(ConcreteOp&,
                  std::vector<std::shared_ptr<OperandImpl>> operands,
                  std::vector<std::shared_ptr<ResultImpl>> results,
                  std::vector<RegionImpl> regions,
                  std::any attributes,
                  std::optional<Location> loc = {})
        : m_type(typeid(ConcreteOp)),
          m_operands(operands),
          m_results(results),
          m_regions(regions),
          m_attributes(attributes),
          m_loc(loc) {}

    std::type_index Type() const { return m_type; }
    std::span<const std::shared_ptr<OperandImpl>> Operands() const { return m_operands; }
    std::span<const std::shared_ptr<ResultImpl>> Results() const { return m_results; }
    std::span<RegionImpl> Regions() { return m_regions; }
    const std::any& Attributes() const { return m_attributes; }
    const std::optional<Location>& Location() const { return m_loc; }

private:
    std::type_index m_type;
    std::vector<std::shared_ptr<OperandImpl>> m_operands;
    std::vector<std::shared_ptr<ResultImpl>> m_results;
    std::vector<RegionImpl> m_regions;
    std::any m_attributes;
    std::optional<dag::Location> m_loc;
};


class Result;
class Operand;
class Operation;


struct Region {
    std::vector<Result> args;
    std::vector<Operation> operations;
};


std::vector<std::shared_ptr<ResultImpl>> AsImpls(std::span<const Result> args);
std::vector<std::shared_ptr<OperandImpl>> AsImpls(std::span<const Operand> args);
std::vector<RegionImpl> AsImpls(std::span<const Region> args);


class Result {
public:
    Result(Operation def, size_t index);
    Result(std::shared_ptr<ResultImpl> impl) : impl(impl) {}
    operator std::shared_ptr<ResultImpl>() const { return impl; }
    ResultImpl* operator->() const { return impl.get(); }

private:
    std::shared_ptr<ResultImpl> impl;
};


class Operand {
public:
    Operand(Result result, Operation user);
    Operand(std::shared_ptr<OperandImpl> impl) : impl(impl) {}
    operator std::shared_ptr<OperandImpl>() const { return impl; }
    OperandImpl* operator->() const { return impl.get(); }

private:
    std::shared_ptr<OperandImpl> impl;
};


class Operation {
public:
    template <class ConcreteOp>
    Operation(ConcreteOp& op,
              std::span<const Operand> operands,
              std::span<const Result> results,
              std::span<const Region> regions,
              std::any attributes,
              std::optional<const Location> loc = {})
        : impl(std::make_shared<OperationImpl>(op, AsImpls(operands), AsImpls(results), AsImpls(regions), attributes, loc)) {}

    operator std::shared_ptr<OperationImpl>() const { return impl; }
    OperationImpl* operator->() const { return impl.get(); }

private:
    std::shared_ptr<OperationImpl> impl;
};


} // namespace dag