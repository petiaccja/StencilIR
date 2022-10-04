#pragma once

#include "ASTTypes.hpp"

#include <array>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace ast {

//------------------------------------------------------------------------------
// Basics
//------------------------------------------------------------------------------

struct Location {
    std::string file;
    int line;
    int col;
};


struct Parameter {
    std::string name;
    types::Type type;
};


struct Node {
    virtual ~Node() = default;
    Node(std::optional<Location> loc = {}) : location(loc) {}
    std::optional<Location> location;
};


struct Statement : Node {
    using Node::Node;
};


struct Expression : Statement {
    using Statement::Statement;
};


//------------------------------------------------------------------------------
// Symbols
//------------------------------------------------------------------------------

struct SymbolRef : Expression {
    explicit SymbolRef(std::string name, std::optional<Location> loc = {})
        : Expression(loc), name(name) {}
    std::string name;
};


struct Assign : Statement {
    explicit Assign(std::vector<std::string> names,
                    std::shared_ptr<Expression> expr,
                    std::optional<Location> loc = {})
        : Statement(loc), names(names), expr(expr) {}
    std::vector<std::string> names;
    std::shared_ptr<Expression> expr;
};


//------------------------------------------------------------------------------
// Stencil structure
//------------------------------------------------------------------------------

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


struct Apply : Expression {
    explicit Apply(std::string callee,
                   std::vector<std::shared_ptr<Expression>> arguments,
                   std::vector<std::shared_ptr<Expression>> targets,
                   std::vector<std::shared_ptr<Expression>> offsets,
                   std::optional<Location> loc = {})
        : Expression(loc), callee(callee), inputs(arguments), outputs(targets), offsets(offsets), static_offsets() {}
    explicit Apply(std::string callee,
                   std::vector<std::shared_ptr<Expression>> arguments,
                   std::vector<std::shared_ptr<Expression>> targets,
                   std::vector<int64_t> static_offsets,
                   std::optional<Location> loc = {})
        : Expression(loc), callee(callee), inputs(arguments), outputs(targets), offsets(), static_offsets(static_offsets) {}
    std::string callee;
    std::vector<std::shared_ptr<Expression>> inputs;
    std::vector<std::shared_ptr<Expression>> outputs;
    std::vector<std::shared_ptr<Expression>> offsets;
    std::vector<int64_t> static_offsets;
};


struct Return : Statement {
    explicit Return(std::vector<std::shared_ptr<Expression>> values = {},
                    std::optional<Location> loc = {})
        : Statement(loc), values(values) {}
    std::vector<std::shared_ptr<Expression>> values;
};


//------------------------------------------------------------------------------
// Module structure
//------------------------------------------------------------------------------

struct Function : Node {
    explicit Function(std::string name,
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


struct Module : Node {
    explicit Module(std::vector<std::shared_ptr<Function>> functions = {},
                    std::vector<std::shared_ptr<Stencil>> stencils = {},
                    std::optional<Location> loc = {})
        : Node(loc), stencils(stencils), functions(functions) {}
    std::vector<std::shared_ptr<Function>> functions;
    std::vector<std::shared_ptr<Stencil>> stencils;
};


//------------------------------------------------------------------------------
// Stencil intrinsics
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

struct For : Expression {
    explicit For(std::shared_ptr<Expression> start,
                 std::shared_ptr<Expression> end,
                 int64_t step,
                 std::string loopVarSymbol,
                 std::vector<std::shared_ptr<Statement>> body,
                 std::vector<std::shared_ptr<Expression>> initArgs,
                 std::vector<std::string> iterArgSymbols,
                 std::optional<Location> loc = {})
        : Expression(loc), start(start), end(end), step(step), loopVarSymbol(loopVarSymbol), body(body), initArgs(initArgs), iterArgSymbols(iterArgSymbols) {}
    std::shared_ptr<Expression> start;
    std::shared_ptr<Expression> end;
    int64_t step;
    std::string loopVarSymbol;
    std::vector<std::shared_ptr<Statement>> body;
    std::vector<std::shared_ptr<Expression>> initArgs;
    std::vector<std::string> iterArgSymbols;
};

struct If : Expression {
    explicit If(std::shared_ptr<Expression> condition,
                std::vector<std::shared_ptr<Statement>> bodyTrue,
                std::vector<std::shared_ptr<Statement>> bodyFalse,
                std::optional<Location> loc = {})
        : Expression(loc), condition(condition), thenBody(bodyTrue), elseBody(bodyFalse) {}
    std::shared_ptr<Expression> condition;
    std::vector<std::shared_ptr<Statement>> thenBody;
    std::vector<std::shared_ptr<Statement>> elseBody;
};

struct Yield : Statement {
    explicit Yield(std::vector<std::shared_ptr<Expression>> values = {},
                   std::optional<Location> loc = {})
        : Statement(loc), values(values) {}
    std::vector<std::shared_ptr<Expression>> values;
};


//------------------------------------------------------------------------------
// Tensors
//------------------------------------------------------------------------------

struct AllocTensor : Expression {
    AllocTensor(types::FundamentalType elementType,
                std::vector<std::shared_ptr<Expression>> sizes,
                Location loc = {})
        : Expression(loc), elementType(elementType), sizes(sizes) {}

    types::FundamentalType elementType;
    std::vector<std::shared_ptr<Expression>> sizes;
};

struct Dim : Expression {
    Dim(std::shared_ptr<Expression> field,
        std::shared_ptr<Expression> index,
        Location loc = {})
        : Expression(loc), field(field), index(index) {}
    std::shared_ptr<Expression> field;
    std::shared_ptr<Expression> index;
};


//------------------------------------------------------------------------------
// Arithmetic-logic
//------------------------------------------------------------------------------

template <class T>
struct Constant : Expression {
    explicit Constant(T value, std::optional<Location> loc = {})
        : Expression(loc), value(value) {}
    Constant(T value, types::Type type, std::optional<Location> loc = {})
        : Expression(loc), value(value), type(type) {}
    T value;
    std::optional<types::Type> type;
};

struct BinaryOperator : Expression {
    BinaryOperator(std::shared_ptr<Expression> lhs,
                   std::shared_ptr<Expression> rhs,
                   std::optional<Location> loc = {}) : Expression(loc), lhs(lhs), rhs(rhs) {}
    std::shared_ptr<Expression> lhs;
    std::shared_ptr<Expression> rhs;
};

struct BinaryArithmeticOperator : BinaryOperator {
    enum eOperation {
        ADD,
        SUB,
        MUL,
        DIV,
        MOD,
        BIT_AND,
        BIT_OR,
        BIT_XOR,
        BIT_SHL,
        BIT_SHR,
    };
    BinaryArithmeticOperator(std::shared_ptr<Expression> lhs,
                             std::shared_ptr<Expression> rhs,
                             eOperation operation,
                             std::optional<Location> loc = {})
        : BinaryOperator(lhs, rhs, loc), operation(operation) {}
    eOperation operation;
};

struct BinaryComparisonOperator : BinaryOperator {
    enum eOperation {
        EQ,
        NEQ,
        LT,
        GT,
        LTE,
        GTE,
    };
    BinaryComparisonOperator(std::shared_ptr<Expression> lhs,
                             std::shared_ptr<Expression> rhs,
                             eOperation operation,
                             std::optional<Location> loc = {})
        : BinaryOperator(lhs, rhs, loc), operation(operation) {}
    eOperation operation;
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