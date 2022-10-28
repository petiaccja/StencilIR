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
    Type type;
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


//------------------------------------------------------------------------------
// Stencil structure
//------------------------------------------------------------------------------

struct Stencil : Node {
    explicit Stencil(std::string name,
                     std::vector<Parameter> parameters,
                     std::vector<Type> results,
                     std::vector<std::shared_ptr<Statement>> body,
                     size_t numDimensions,
                     std::optional<Location> loc = {})
        : Node(loc), name(name), parameters(parameters), results(results), body(body), numDimensions(numDimensions) {}
    std::string name;
    std::vector<Parameter> parameters;
    std::vector<Type> results;
    std::vector<std::shared_ptr<Statement>> body;
    size_t numDimensions;
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
                      std::vector<Type> results,
                      std::vector<std::shared_ptr<Statement>> body,
                      std::optional<Location> loc = {})
        : Node(loc), name(name), parameters(parameters), results(results), body(body) {}
    std::string name;
    std::vector<Parameter> parameters;
    std::vector<Type> results;
    std::vector<std::shared_ptr<Statement>> body;
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
        : Node(loc), functions(functions), stencils(stencils){}
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

//------------------------------------------------------------------------------
// Control flow
//------------------------------------------------------------------------------

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
    AllocTensor(ScalarType elementType,
                std::vector<std::shared_ptr<Expression>> sizes,
                std::optional<Location> loc = {})
        : Expression(loc), elementType(elementType), sizes(sizes) {}

    ScalarType elementType;
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
    explicit Constant(std::integral auto value, std::optional<Location> loc = {})
        : Expression(loc), value(value), type(InferType<decltype(value)>()) {}
    explicit Constant(std::floating_point auto value, std::optional<Location> loc = {})
        : Expression(loc), value(value), type(InferType<decltype(value)>()) {}
    explicit Constant(bool value, std::optional<Location> loc = {})
        : Expression(loc), value(value), type(InferType<bool>()) {}
    explicit Constant(impl::IndexType, ptrdiff_t value, std::optional<Location> loc = {})
        : Expression(loc), value(value), type(ScalarType::INDEX) {}
    std::any value;
    ScalarType type;
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

struct Cast : Expression {
    Cast(std::shared_ptr<Expression> expr,
         Type type) : expr(expr), type(type) {}
    std::shared_ptr<Expression> expr;
    Type type;
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