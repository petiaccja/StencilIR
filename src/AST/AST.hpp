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

struct Stencil : Node {
    explicit Stencil(std::string name,
                        std::vector<Parameter> parameters,
                        std::vector<types::Type> results,
                        std::vector<std::shared_ptr<Statement>> body,
                        size_t numDimensions,
                        std::optional<Location> loc = {})
        : Node(loc), name(name), parameters(parameters), results(results), body(body), numDimensions(numDimensions) {}
    std::string name;
    std::vector<Parameter> parameters;
    std::vector<types::Type> results;
    std::vector<std::shared_ptr<Statement>> body;
    size_t numDimensions;
};

struct Return : Statement {
    explicit Return(std::vector<std::shared_ptr<Expression>> values = {},
                          std::optional<Location> loc = {})
        : Statement(loc), values(values) {}
    std::vector<std::shared_ptr<Expression>> values;
};

struct Apply : Statement {
    explicit Apply(std::string callee,
                          std::vector<std::shared_ptr<Expression>> gridDim,
                          std::vector<std::shared_ptr<Expression>> arguments = {},
                          std::vector<std::shared_ptr<Expression>> targets = {},
                          std::optional<Location> loc = {})
        : Statement(loc), callee(callee), gridDim(gridDim), arguments(arguments), targets(targets) {}
    std::string callee;
    std::vector<std::shared_ptr<Expression>> gridDim;
    std::vector<std::shared_ptr<Expression>> arguments;
    std::vector<std::shared_ptr<Expression>> targets;
};

struct Module : Node {
    explicit Module(std::vector<std::shared_ptr<Node>> body,
                    std::vector<std::shared_ptr<Stencil>> kernels = {},
                    std::vector<Parameter> parameters = {},
                    std::optional<Location> loc = {})
        : Node(loc), body(body), kernels(kernels), parameters(parameters) {}
    std::vector<std::shared_ptr<Node>> body;
    std::vector<std::shared_ptr<Stencil>> kernels;
    std::vector<Parameter> parameters;
};

//------------------------------------------------------------------------------
// Kernel intrinsics
//------------------------------------------------------------------------------

struct Index : Expression {
    explicit Index(std::optional<Location> loc = {}) : Expression(loc) {}
};

struct Jump : Expression {
    explicit Jump(std::shared_ptr<Expression> index,
                  std::vector<int64_t> offset,
                  std::optional<Location> loc = {})
        : Expression(loc), index(index), offset(offset) {}
    std::shared_ptr<Expression> index;
    std::vector<int64_t> offset;
};

struct Sample : Expression {
    explicit Sample(std::shared_ptr<Expression> field,
                    std::shared_ptr<Expression> index,
                    std::optional<Location> loc = {})
        : Expression(loc), field(field), index(index) {}
    std::shared_ptr<Expression> field;
    std::shared_ptr<Expression> index;
};

struct JumpIndirect : Expression {
    explicit JumpIndirect(std::shared_ptr<Expression> index,
                          int64_t dimension,
                          std::shared_ptr<Expression> map,
                          std::shared_ptr<Expression> mapElement,
                          std::optional<Location> loc = {})
        : Expression(loc), index(index), dimension(dimension), map(map), mapElement(mapElement) {}
    std::shared_ptr<Expression> index;
    int64_t dimension;
    std::shared_ptr<Expression> map;
    std::shared_ptr<Expression> mapElement;
};

struct SampleIndirect : Expression {
    explicit SampleIndirect(std::shared_ptr<Expression> index,
                            int64_t dimension,
                            std::shared_ptr<Expression> field,
                            std::shared_ptr<Expression> fieldElement,
                            std::optional<Location> loc = {})
        : Expression(loc), index(index), dimension(dimension), field(field), fieldElement(fieldElement) {}
    std::shared_ptr<Expression> index;
    int64_t dimension;
    std::shared_ptr<Expression> field;
    std::shared_ptr<Expression> fieldElement;
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

struct Sub : Expression {
    Sub(std::shared_ptr<Expression> lhs, std::shared_ptr<Expression> rhs, std::optional<Location> loc = {})
        : Expression(loc), lhs(lhs), rhs(rhs) {}
    std::shared_ptr<Expression> lhs;
    std::shared_ptr<Expression> rhs;
};

struct Mul : Expression {
    Mul(std::shared_ptr<Expression> lhs, std::shared_ptr<Expression> rhs, std::optional<Location> loc = {})
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