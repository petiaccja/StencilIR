#pragma once

#include "Node.hpp"
#include <memory>
#include <optional>
#include <utility>

namespace ast {


struct Statement : Node {
    using Node::Node;
};

struct Expression : Statement {
    using Statement::Statement;
};

template <class T>
struct Constant : Expression {
    explicit Constant(T value, std::optional<Location> loc = {})
        : Expression(location), value(value) {}
    T value;
};

struct Add : Expression {
    Add(std::shared_ptr<Expression> lhs, std::shared_ptr<Expression> rhs, std::optional<Location> loc = {})
        : Expression(loc), lhs(lhs), rhs(rhs) {}
    std::shared_ptr<Expression> lhs;
    std::shared_ptr<Expression> rhs;
};

struct Print : Statement {
    explicit Print(std::shared_ptr<Expression> argument, std::optional<Location> loc = {})
        : Statement(loc), argument(argument) {}
    std::shared_ptr<Expression> argument;
};

struct Module : Node {
    explicit Module(std::vector<std::shared_ptr<Statement>> body, std::optional<Location> loc = {})
        : Node(loc), body(body) {}
    std::vector<std::shared_ptr<Statement>> body;
};

using StatementList = std::vector<std::shared_ptr<Statement>>;


} // namespace ast