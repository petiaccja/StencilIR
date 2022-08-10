#pragma once

#include "Node.hpp"
#include "Types.hpp"
#include <array>
#include <memory>
#include <optional>
#include <utility>

namespace ast {

//------------------------------------------------------------------------------
// Basics
//------------------------------------------------------------------------------

struct Statement : Node {
    using Node::Node;
};

struct Expression : Statement {
    using Statement::Statement;
};


//------------------------------------------------------------------------------
// Structure
//------------------------------------------------------------------------------

struct SymbolRef : Expression {
    explicit SymbolRef(std::string name, std::optional<Location> loc = {})
        : Expression(loc), name(name) {}
    std::string name;
};

struct Parameter {
    std::string name;
    types::Type type;
};

struct KernelFunc : Node {
    explicit KernelFunc(std::string name,
                        std::vector<Parameter> parameters,
                        std::vector<types::Type> results,
                        std::vector<std::shared_ptr<Statement>> body,
                        std::optional<Location> loc = {})
        : Node(loc), name(name), parameters(parameters), results(results), body(body) {}
    std::string name;
    std::vector<Parameter> parameters;
    std::vector<types::Type> results;
    std::vector<std::shared_ptr<Statement>> body;
};

struct KernelReturn : Statement {
    explicit KernelReturn(std::vector<std::shared_ptr<Expression>> values = {},
                          std::optional<Location> loc = {})
        : Statement(loc), values(values) {}
    std::vector<std::shared_ptr<Expression>> values;
};

struct KernelLaunch : Statement {
    explicit KernelLaunch(std::string callee,
                          std::vector<std::shared_ptr<Expression>> gridDim,
                          std::vector<std::shared_ptr<Expression>> arguments,
                          std::optional<Location> loc = {})
        : Statement(loc), callee(callee), gridDim(gridDim), arguments(arguments) {}
    std::string callee;
    std::vector<std::shared_ptr<Expression>> gridDim;
    std::vector<std::shared_ptr<Expression>> arguments;
};

struct Module : Node {
    explicit Module(std::vector<std::shared_ptr<Node>> body,
                    std::vector<std::shared_ptr<KernelFunc>> kernels = {},
                    std::vector<Parameter> parameters = {},
                    std::optional<Location> loc = {})
        : Node(loc), body(body), kernels(kernels), parameters(parameters) {}
    std::vector<std::shared_ptr<Node>> body;
    std::vector<std::shared_ptr<KernelFunc>> kernels;
    std::vector<Parameter> parameters;
};


//------------------------------------------------------------------------------
// Arithmetic-logic
//------------------------------------------------------------------------------

template <class T>
struct Constant : Expression {
    explicit Constant(T value, std::optional<Location> loc = {})
        : Expression(location), value(value) {}
    Constant(T value, types::Type type, std::optional<Location> loc = {})
        : Expression(location), value(value), type(type) {}
    T value;
    std::optional<types::Type> type;
};

struct Add : Expression {
    Add(std::shared_ptr<Expression> lhs, std::shared_ptr<Expression> rhs, std::optional<Location> loc = {})
        : Expression(loc), lhs(lhs), rhs(rhs) {}
    std::shared_ptr<Expression> lhs;
    std::shared_ptr<Expression> rhs;
};

struct ReshapeField : Expression {
    explicit ReshapeField(std::shared_ptr<Expression> field,
                          std::vector<std::shared_ptr<Expression>> shape,
                          std::vector<std::shared_ptr<Expression>> strides,
                          Location loc = {})
        : Expression(loc), field(field), shape(shape), strides(strides) {}
    std::shared_ptr<Expression> field;
    std::vector<std::shared_ptr<Expression>> shape;
    std::vector<std::shared_ptr<Expression>> strides;
};

//------------------------------------------------------------------------------
// Misc
//------------------------------------------------------------------------------

struct Print : Statement {
    explicit Print(std::shared_ptr<Expression> argument, std::optional<Location> loc = {})
        : Statement(loc), argument(argument) {}
    std::shared_ptr<Expression> argument;
};


} // namespace ast