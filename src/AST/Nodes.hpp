#pragma once

#include "Types.hpp"

#include <any>
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
    TypePtr type;
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
                    std::vector<std::shared_ptr<Expression>> exprs,
                    std::optional<Location> loc = {})
        : Statement(loc), names(names), exprs(exprs) {}
    std::vector<std::string> names;
    std::vector<std::shared_ptr<Expression>> exprs;
};


struct Pack : Expression {
    explicit Pack(std::vector<std::shared_ptr<Expression>> exprs,
                  std::optional<Location> loc = {})
        : Expression(loc), exprs(exprs) {}
    std::vector<std::shared_ptr<Expression>> exprs;
};


//------------------------------------------------------------------------------
// Stencil structure
//------------------------------------------------------------------------------

struct Stencil : Node {
    explicit Stencil(std::string name,
                     std::vector<Parameter> parameters,
                     std::vector<TypePtr> results,
                     std::vector<std::shared_ptr<Statement>> body,
                     size_t numDimensions,
                     bool isPublic = false,
                     std::optional<Location> loc = {})
        : Node(loc), name(name), parameters(parameters), results(results), body(body), numDimensions(numDimensions), isPublic(isPublic) {}
    std::string name;
    std::vector<Parameter> parameters;
    std::vector<TypePtr> results;
    std::vector<std::shared_ptr<Statement>> body;
    size_t numDimensions;
    bool isPublic;
};


struct Apply : Expression {
    explicit Apply(std::string callee,
                   std::vector<std::shared_ptr<Expression>> inputs,
                   std::vector<std::shared_ptr<Expression>> outputs,
                   std::vector<std::shared_ptr<Expression>> offsets,
                   std::optional<Location> loc = {})
        : Apply(callee, inputs, outputs, offsets, {}, loc) {}
    explicit Apply(std::string callee,
                   std::vector<std::shared_ptr<Expression>> inputs,
                   std::vector<std::shared_ptr<Expression>> outputs,
                   std::vector<int64_t> static_offsets,
                   std::optional<Location> loc = {})
        : Apply(callee, inputs, outputs, {}, static_offsets, loc) {}
    explicit Apply(std::string callee,
                   std::vector<std::shared_ptr<Expression>> inputs,
                   std::vector<std::shared_ptr<Expression>> outputs,
                   std::vector<std::shared_ptr<Expression>> offsets,
                   std::vector<int64_t> static_offsets,
                   std::optional<Location> loc = {})
        : Expression(loc), callee(callee), inputs(inputs), outputs(outputs), offsets(offsets), static_offsets(static_offsets) {}
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
                      std::vector<TypePtr> results,
                      std::vector<std::shared_ptr<Statement>> body,
                      bool isPublic = true,
                      std::optional<Location> loc = {})
        : Node(loc), name(name), parameters(parameters), results(results), body(body), isPublic(isPublic) {}
    std::string name;
    std::vector<Parameter> parameters;
    std::vector<TypePtr> results;
    std::vector<std::shared_ptr<Statement>> body;
    bool isPublic;
};


struct Call : Expression {
    explicit Call(std::string callee,
                  std::vector<std::shared_ptr<Expression>> args,
                  std::optional<Location> loc = {})
        : Expression(loc), callee(callee), args(args) {}
    std::string callee;
    std::vector<std::shared_ptr<Expression>> args;
};


struct Module : Node {
    explicit Module(std::vector<std::shared_ptr<Function>> functions = {},
                    std::vector<std::shared_ptr<Stencil>> stencils = {},
                    std::optional<Location> loc = {})
        : Node(loc), functions(functions), stencils(stencils) {}
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

struct Project : Expression {
    explicit Project(std::shared_ptr<Expression> index,
                     std::vector<int64_t> positions,
                     std::optional<Location> loc = {})
        : Expression(loc), index(index), positions(positions) {}
    std::shared_ptr<Expression> index;
    std::vector<int64_t> positions;
};

struct Extend : Expression {
    explicit Extend(std::shared_ptr<Expression> index,
                    int64_t position,
                    std::shared_ptr<Expression> value,
                    std::optional<Location> loc = {})
        : Expression(loc), index(index), position(position), value(value) {}
    std::shared_ptr<Expression> index;
    int64_t position;
    std::shared_ptr<Expression> value;
};

struct Exchange : Expression {
    explicit Exchange(std::shared_ptr<Expression> index,
                      int64_t position,
                      std::shared_ptr<Expression> value,
                      std::optional<Location> loc = {})
        : Expression(loc), index(index), position(position), value(value) {}
    std::shared_ptr<Expression> index;
    int64_t position;
    std::shared_ptr<Expression> value;
};

struct Extract : Expression {
    explicit Extract(std::shared_ptr<Expression> index,
                     int64_t position,
                     std::optional<Location> loc = {})
        : Expression(loc), index(index), position(position) {}
    std::shared_ptr<Expression> index;
    int64_t position;
};

struct Sample : Expression {
    explicit Sample(std::shared_ptr<Expression> field,
                    std::shared_ptr<Expression> index,
                    std::optional<Location> loc = {})
        : Expression(loc), field(field), index(index) {}
    std::shared_ptr<Expression> field;
    std::shared_ptr<Expression> index;
};


//------------------------------------------------------------------------------
// Control flow
//------------------------------------------------------------------------------

struct For : Expression {
    explicit For(std::shared_ptr<Expression> start,
                 std::shared_ptr<Expression> end,
                 std::shared_ptr<Expression> step,
                 std::string loopVarSymbol,
                 std::vector<std::shared_ptr<Statement>> body,
                 std::vector<std::shared_ptr<Expression>> initArgs,
                 std::vector<std::string> iterArgSymbols,
                 std::optional<Location> loc = {})
        : Expression(loc), start(start), end(end), step(step), loopVarSymbol(loopVarSymbol), body(body), initArgs(initArgs), iterArgSymbols(iterArgSymbols) {}
    std::shared_ptr<Expression> start;
    std::shared_ptr<Expression> end;
    std::shared_ptr<Expression> step;
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

struct Block : Expression {
    explicit Block(std::vector<std::shared_ptr<Statement>> body,
                   std::optional<Location> loc = {})
        : Expression(loc), body(body) {}
    std::vector<std::shared_ptr<Statement>> body;
};

//------------------------------------------------------------------------------
// Tensors
//------------------------------------------------------------------------------

struct AllocTensor : Expression {
    AllocTensor(TypePtr elementType,
                std::vector<std::shared_ptr<Expression>> sizes,
                std::optional<Location> loc = {})
        : Expression(loc), elementType(elementType), sizes(sizes) {}

    TypePtr elementType;
    std::vector<std::shared_ptr<Expression>> sizes;
};

struct Dim : Expression {
    Dim(std::shared_ptr<Expression> field,
        std::shared_ptr<Expression> index,
        std::optional<Location> loc = {})
        : Expression(loc), field(field), index(index) {}
    std::shared_ptr<Expression> field;
    std::shared_ptr<Expression> index;
};

struct ExtractSlice : Expression {
    ExtractSlice(std::shared_ptr<Expression> source,
                 std::vector<std::shared_ptr<Expression>> offsets,
                 std::vector<std::shared_ptr<Expression>> sizes,
                 std::vector<std::shared_ptr<Expression>> strides,
                 std::optional<Location> loc = {})
        : Expression(loc), source(source), offsets(offsets), sizes(sizes), strides(strides) {}
    std::shared_ptr<Expression> source;
    std::vector<std::shared_ptr<Expression>> offsets;
    std::vector<std::shared_ptr<Expression>> sizes;
    std::vector<std::shared_ptr<Expression>> strides;
};

struct InsertSlice : Expression {
    InsertSlice(std::shared_ptr<Expression> source,
                std::shared_ptr<Expression> dest,
                std::vector<std::shared_ptr<Expression>> offsets,
                std::vector<std::shared_ptr<Expression>> sizes,
                std::vector<std::shared_ptr<Expression>> strides,
                std::optional<Location> loc = {})
        : Expression(loc), source(source), dest(dest), offsets(offsets), sizes(sizes), strides(strides) {}
    std::shared_ptr<Expression> source;
    std::shared_ptr<Expression> dest;
    std::vector<std::shared_ptr<Expression>> offsets;
    std::vector<std::shared_ptr<Expression>> sizes;
    std::vector<std::shared_ptr<Expression>> strides;
};

//------------------------------------------------------------------------------
// Arithmetic-logic
//------------------------------------------------------------------------------

struct Constant : Expression {
    template <class T>
    explicit Constant(T value, std::optional<Location> loc = {})
        : Constant(std::move(value), InferType<std::decay_t<T>>(), std::move(loc)) {}
    explicit Constant(auto value, TypePtr type, std::optional<Location> loc = {})
        : Expression(loc), type(type) {
        if (std::dynamic_pointer_cast<IntegerType>(type)) {
            this->value = static_cast<int64_t>(value);
        }
        else if (std::dynamic_pointer_cast<IndexType>(type)) {
            this->value = static_cast<int64_t>(value);
        }
        else if (std::dynamic_pointer_cast<FloatType>(type)) {
            this->value = static_cast<double>(value);
        }
        else {
            this->value = value;
        }
    }
    std::any value;
    TypePtr type;
};

struct BinaryOperator : Expression {
    BinaryOperator(std::shared_ptr<Expression> lhs,
                   std::shared_ptr<Expression> rhs,
                   std::optional<Location> loc = {}) : Expression(loc), lhs(lhs), rhs(rhs) {}
    std::shared_ptr<Expression> lhs;
    std::shared_ptr<Expression> rhs;
};

enum class eArithmeticFunction {
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

struct ArithmeticOperator : BinaryOperator {
    ArithmeticOperator(std::shared_ptr<Expression> lhs,
                       std::shared_ptr<Expression> rhs,
                       eArithmeticFunction operation,
                       std::optional<Location> loc = {})
        : BinaryOperator(lhs, rhs, loc), operation(operation) {}
    eArithmeticFunction operation;
};

enum class eComparisonFunction {
    EQ,
    NEQ,
    LT,
    GT,
    LTE,
    GTE,
};

struct ComparisonOperator : BinaryOperator {
    ComparisonOperator(std::shared_ptr<Expression> lhs,
                       std::shared_ptr<Expression> rhs,
                       eComparisonFunction operation,
                       std::optional<Location> loc = {})
        : BinaryOperator(lhs, rhs, loc), operation(operation) {}
    eComparisonFunction operation;
};

struct Min : Expression {
    Min(std::shared_ptr<Expression> lhs,
        std::shared_ptr<Expression> rhs,
        std::optional<Location> loc = {})
        : Expression(loc), lhs(lhs), rhs(rhs) {}
    std::shared_ptr<Expression> lhs;
    std::shared_ptr<Expression> rhs;
};

struct Max : Expression {
    Max(std::shared_ptr<Expression> lhs,
        std::shared_ptr<Expression> rhs,
        std::optional<Location> loc = {})
        : Expression(loc), lhs(lhs), rhs(rhs) {}
    std::shared_ptr<Expression> lhs;
    std::shared_ptr<Expression> rhs;
};

struct Cast : Expression {
    Cast(std::shared_ptr<Expression> expr,
         TypePtr type,
         std::optional<Location> loc = {})
        : Expression(loc), expr(expr), type(type) {}
    std::shared_ptr<Expression> expr;
    TypePtr type;
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