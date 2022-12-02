#pragma once


#include "Nodes.hpp"
#include "Types.hpp"


namespace ast {



//------------------------------------------------------------------------------
// Symbols
//------------------------------------------------------------------------------

inline auto symref(std::string name,
                   std::optional<Location> loc = {}) {
    return std::make_shared<SymbolRef>(name, loc);
}


inline auto assign(std::vector<std::string> names,
                   std::vector<std::shared_ptr<Expression>> exprs,
                   std::optional<Location> loc = {}) {
    return std::make_shared<Assign>(names, exprs, loc);
}


inline auto assign(std::string name,
                   std::shared_ptr<Expression> expr,
                   std::optional<Location> loc = {}) {
    return assign(std::vector{ name }, std::vector{ expr }, loc);
}


inline auto pack(std::vector<std::shared_ptr<Expression>> exprs,
                 std::optional<Location> loc = {}) {
    return std::make_shared<Pack>(exprs, loc);
}


//------------------------------------------------------------------------------
// Stencil structure
//------------------------------------------------------------------------------

inline auto stencil(std::string name,
                    std::vector<Parameter> parameters,
                    std::vector<Type> results,
                    std::vector<std::shared_ptr<Statement>> body,
                    size_t numDimensions,
                    std::optional<Location> loc = {}) {
    return std::make_shared<Stencil>(name, parameters, results, body, numDimensions, loc);
}


inline auto return_(std::vector<std::shared_ptr<Expression>> values = {},
                    std::optional<Location> loc = {}) {
    return std::make_shared<Return>(values, loc);
}


inline auto apply(std::string callee,
                  std::vector<std::shared_ptr<Expression>> inputs,
                  std::vector<std::shared_ptr<Expression>> outputs,
                  std::vector<std::shared_ptr<Expression>> offsets,
                  std::optional<Location> loc = {}) {
    return std::make_shared<Apply>(callee, inputs, outputs, offsets, loc);
}


inline auto apply(std::string callee,
                  std::vector<std::shared_ptr<Expression>> inputs,
                  std::vector<std::shared_ptr<Expression>> outputs,
                  std::vector<int64_t> static_offsets = {},
                  std::optional<Location> loc = {}) {
    return std::make_shared<Apply>(callee, inputs, outputs, static_offsets, loc);
}


//------------------------------------------------------------------------------
// Module structure
//------------------------------------------------------------------------------

inline auto function(std::string name,
                     std::vector<Parameter> parameters,
                     std::vector<Type> results,
                     std::vector<std::shared_ptr<Statement>> body,
                     std::optional<Location> loc = {}) {
    return std::make_shared<Function>(name, parameters, results, body, loc);
}


inline auto call(std::string name,
                 std::vector<std::shared_ptr<Expression>> args,
                 std::optional<Location> loc = {}) {
    return std::make_shared<Call>(name, args, loc);
}


inline auto module_(std::vector<std::shared_ptr<Function>> functions = {},
                    std::vector<std::shared_ptr<Stencil>> stencils = {},
                    std::optional<Location> loc = {}) {
    return std::make_shared<Module>(functions, stencils, loc);
}


//------------------------------------------------------------------------------
// Stencil intrinsics
//------------------------------------------------------------------------------

inline auto index(std::optional<Location> loc = {}) {
    return std::make_shared<Index>();
}


inline auto jump(std::shared_ptr<Expression> index,
                 std::vector<int64_t> offset,
                 std::optional<Location> loc = {}) {
    return std::make_shared<Jump>(index, offset, loc);
}


inline auto project(std::shared_ptr<Expression> index,
                    std::vector<int64_t> positions,
                    std::optional<Location> loc = {}) {
    return std::make_shared<Project>(index, positions, loc);
}


inline auto extend(std::shared_ptr<Expression> index,
                   int64_t position,
                   std::shared_ptr<Expression> value,
                   std::optional<Location> loc = {}) {
    return std::make_shared<Extend>(index, position, value, loc);
}


inline auto exchange(std::shared_ptr<Expression> index,
                     int64_t position,
                     std::shared_ptr<Expression> value,
                     std::optional<Location> loc = {}) {
    return std::make_shared<Exchange>(index, position, value, loc);
}


inline auto extract(std::shared_ptr<Expression> index,
                    int64_t position,
                    std::optional<Location> loc = {}) {
    return std::make_shared<Extract>(index, position, loc);
}


inline auto jump_indirect(std::shared_ptr<Expression> index,
                          int64_t dimension,
                          std::shared_ptr<Expression> map,
                          std::shared_ptr<Expression> mapElement,
                          std::optional<Location> loc = {}) {
    return std::make_shared<JumpIndirect>(index, dimension, map, mapElement, loc);
}


inline auto sample(std::shared_ptr<Expression> field,
                   std::shared_ptr<Expression> index,
                   std::optional<Location> loc = {}) {
    return std::make_shared<Sample>(field, index, loc);
}


inline auto sample_indirect(std::shared_ptr<Expression> index,
                            int64_t dimension,
                            std::shared_ptr<Expression> field,
                            std::shared_ptr<Expression> fieldElement,
                            std::optional<Location> loc = {}) {
    return std::make_shared<SampleIndirect>(index, dimension, field, fieldElement, loc);
}


//--------------------------------------------------------------------------
// Structured flow control
//--------------------------------------------------------------------------

inline auto for_(std::shared_ptr<Expression> start,
                 std::shared_ptr<Expression> end,
                 std::shared_ptr<Expression> step,
                 std::string loopVarSymbol,
                 std::vector<std::shared_ptr<Statement>> body,
                 std::vector<std::shared_ptr<Expression>> initArgs,
                 std::vector<std::string> iterArgSymbols,
                 std::optional<Location> loc = {}) {
    return std::make_shared<For>(start, end, step, loopVarSymbol, body, initArgs, iterArgSymbols);
};


inline auto if_(std::shared_ptr<Expression> condition,
                std::vector<std::shared_ptr<Statement>> bodyTrue,
                std::vector<std::shared_ptr<Statement>> bodyFalse = {},
                std::optional<Location> loc = {}) {
    return std::make_shared<If>(condition, bodyTrue, bodyFalse);
};


inline auto yield(std::vector<std::shared_ptr<Expression>> values = {},
                  std::optional<Location> loc = {}) {
    return std::make_shared<Yield>(values, loc);
}


inline auto block(std::vector<std::shared_ptr<Statement>> body,
                  std::optional<Location> loc = {}) {
    return std::make_shared<Block>(body, loc);
}


//------------------------------------------------------------------------------
// Arithmetic-logic
//------------------------------------------------------------------------------

template <class T>
inline auto constant(T value, std::optional<Location> loc = {}) {
    return std::make_shared<Constant>(value, loc);
}

template <class T>
inline auto constant(impl::IndexType, T value, std::optional<Location> loc = {}) {
    return std::make_shared<Constant>(index_type, value, loc);
}


inline auto add(std::shared_ptr<Expression> lhs, std::shared_ptr<Expression> rhs, std::optional<Location> loc = {}) {
    return std::make_shared<ArithmeticOperator>(lhs, rhs, eArithmeticFunction::ADD, loc);
}
inline auto sub(std::shared_ptr<Expression> lhs, std::shared_ptr<Expression> rhs, std::optional<Location> loc = {}) {
    return std::make_shared<ArithmeticOperator>(lhs, rhs, eArithmeticFunction::SUB, loc);
}
inline auto mul(std::shared_ptr<Expression> lhs, std::shared_ptr<Expression> rhs, std::optional<Location> loc = {}) {
    return std::make_shared<ArithmeticOperator>(lhs, rhs, eArithmeticFunction::MUL, loc);
}
inline auto div(std::shared_ptr<Expression> lhs, std::shared_ptr<Expression> rhs, std::optional<Location> loc = {}) {
    return std::make_shared<ArithmeticOperator>(lhs, rhs, eArithmeticFunction::DIV, loc);
}
inline auto mod(std::shared_ptr<Expression> lhs, std::shared_ptr<Expression> rhs, std::optional<Location> loc = {}) {
    return std::make_shared<ArithmeticOperator>(lhs, rhs, eArithmeticFunction::MOD, loc);
}
inline auto bit_and(std::shared_ptr<Expression> lhs, std::shared_ptr<Expression> rhs, std::optional<Location> loc = {}) {
    return std::make_shared<ArithmeticOperator>(lhs, rhs, eArithmeticFunction::BIT_AND, loc);
}
inline auto bit_or(std::shared_ptr<Expression> lhs, std::shared_ptr<Expression> rhs, std::optional<Location> loc = {}) {
    return std::make_shared<ArithmeticOperator>(lhs, rhs, eArithmeticFunction::BIT_OR, loc);
}
inline auto bit_xor(std::shared_ptr<Expression> lhs, std::shared_ptr<Expression> rhs, std::optional<Location> loc = {}) {
    return std::make_shared<ArithmeticOperator>(lhs, rhs, eArithmeticFunction::BIT_XOR, loc);
}
inline auto bit_shl(std::shared_ptr<Expression> lhs, std::shared_ptr<Expression> rhs, std::optional<Location> loc = {}) {
    return std::make_shared<ArithmeticOperator>(lhs, rhs, eArithmeticFunction::BIT_SHL, loc);
}
inline auto bit_shr(std::shared_ptr<Expression> lhs, std::shared_ptr<Expression> rhs, std::optional<Location> loc = {}) {
    return std::make_shared<ArithmeticOperator>(lhs, rhs, eArithmeticFunction::BIT_SHR, loc);
}

inline auto eq(std::shared_ptr<Expression> lhs, std::shared_ptr<Expression> rhs, std::optional<Location> loc = {}) {
    return std::make_shared<ComparisonOperator>(lhs, rhs, eComparisonFunction::EQ, loc);
}
inline auto neq(std::shared_ptr<Expression> lhs, std::shared_ptr<Expression> rhs, std::optional<Location> loc = {}) {
    return std::make_shared<ComparisonOperator>(lhs, rhs, eComparisonFunction::NEQ, loc);
}
inline auto gt(std::shared_ptr<Expression> lhs, std::shared_ptr<Expression> rhs, std::optional<Location> loc = {}) {
    return std::make_shared<ComparisonOperator>(lhs, rhs, eComparisonFunction::GT, loc);
}
inline auto lt(std::shared_ptr<Expression> lhs, std::shared_ptr<Expression> rhs, std::optional<Location> loc = {}) {
    return std::make_shared<ComparisonOperator>(lhs, rhs, eComparisonFunction::LT, loc);
}
inline auto lte(std::shared_ptr<Expression> lhs, std::shared_ptr<Expression> rhs, std::optional<Location> loc = {}) {
    return std::make_shared<ComparisonOperator>(lhs, rhs, eComparisonFunction::LTE, loc);
}
inline auto gte(std::shared_ptr<Expression> lhs, std::shared_ptr<Expression> rhs, std::optional<Location> loc = {}) {
    return std::make_shared<ComparisonOperator>(lhs, rhs, eComparisonFunction::GTE, loc);
}

inline auto min(std::shared_ptr<Expression> lhs, std::shared_ptr<Expression> rhs, std::optional<Location> loc = {}) {
    return std::make_shared<Min>(lhs, rhs, loc);
}
inline auto max(std::shared_ptr<Expression> lhs, std::shared_ptr<Expression> rhs, std::optional<Location> loc = {}) {
    return std::make_shared<Max>(lhs, rhs, loc);
}

inline auto cast(std::shared_ptr<Expression> expr, Type type, std::optional<Location> loc = {}) {
    return std::make_shared<Cast>(expr, type, loc);
}


//------------------------------------------------------------------------------
// Tensor
//------------------------------------------------------------------------------

inline auto alloc_tensor(ScalarType elementType,
                         std::vector<std::shared_ptr<Expression>> sizes,
                         Location loc = {}) {
    return std::make_shared<AllocTensor>(elementType, sizes, loc);
}


inline auto dim(std::shared_ptr<Expression> field,
                std::shared_ptr<Expression> index,
                Location loc = {}) {
    return std::make_shared<Dim>(field, index, loc);
}

inline auto extract_slice(std::shared_ptr<Expression> source,
                          std::vector<std::shared_ptr<Expression>> offsets,
                          std::vector<std::shared_ptr<Expression>> sizes,
                          std::vector<std::shared_ptr<Expression>> strides,
                          Location loc = {}) {
    return std::make_shared<ExtractSlice>(source, offsets, sizes, strides, loc);
}

inline auto insert_slice(std::shared_ptr<Expression> source,
                         std::shared_ptr<Expression> dest,
                         std::vector<std::shared_ptr<Expression>> offsets,
                         std::vector<std::shared_ptr<Expression>> sizes,
                         std::vector<std::shared_ptr<Expression>> strides,
                         Location loc = {}) {
    return std::make_shared<InsertSlice>(source, dest, offsets, sizes, strides, loc);
}

} // namespace ast