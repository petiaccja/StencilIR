#pragma once


#include "AST.hpp"


namespace ast {



//------------------------------------------------------------------------------
// Structure
//------------------------------------------------------------------------------


inline auto symref(std::string name,
                   std::optional<Location> loc = {}) {
    return std::make_shared<SymbolRef>(name, loc);
}

inline auto stencil(std::string name,
                    std::vector<Parameter> parameters,
                    std::vector<types::Type> results,
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

inline auto module_(std::vector<std::shared_ptr<Node>> body,
                    std::vector<std::shared_ptr<Stencil>> kernels = {},
                    std::vector<Parameter> parameters = {},
                    std::optional<Location> loc = {}) {
    return std::make_shared<Module>(body, kernels, parameters, loc);
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

inline auto sample(std::shared_ptr<Expression> field,
                   std::shared_ptr<Expression> index,
                   std::optional<Location> loc = {}) {
    return std::make_shared<Sample>(field, index, loc);
}

inline auto jump_indirect(std::shared_ptr<Expression> index,
                          int64_t dimension,
                          std::shared_ptr<Expression> map,
                          std::shared_ptr<Expression> mapElement,
                          std::optional<Location> loc = {}) {
    return std::make_shared<JumpIndirect>(index, dimension, map, mapElement, loc);
}

inline auto sample_indirect(std::shared_ptr<Expression> index,
                            int64_t dimension,
                            std::shared_ptr<Expression> field,
                            std::shared_ptr<Expression> fieldElement,
                            std::optional<Location> loc = {}) {
    return std::make_shared<SampleIndirect>(index, dimension, field, fieldElement, loc);
}

//------------------------------------------------------------------------------
// Arithmetic-logic
//------------------------------------------------------------------------------

template <class T>
inline auto constant(T value,
                     std::optional<Location> loc = {}) {
    return std::make_shared<Constant<T>>(value, loc);
}

template <class T>
inline auto constant(T value,
                     types::Type type,
                     std::optional<Location> loc = {}) {
    return std::make_shared<Constant<T>>(value, type, loc);
}

inline auto add(std::shared_ptr<Expression> lhs,
                std::shared_ptr<Expression> rhs,
                std::optional<Location> loc = {}) {
    return std::make_shared<Add>(lhs, rhs, loc);
}

inline auto sub(std::shared_ptr<Expression> lhs,
                std::shared_ptr<Expression> rhs,
                std::optional<Location> loc = {}) {
    return std::make_shared<Sub>(lhs, rhs, loc);
}

inline auto mul(std::shared_ptr<Expression> lhs,
                std::shared_ptr<Expression> rhs,
                std::optional<Location> loc = {}) {
    return std::make_shared<Mul>(lhs, rhs, loc);
}


//------------------------------------------------------------------------------
// Tensor
//------------------------------------------------------------------------------

inline auto alloc_tensor(types::FundamentalType elementType,
                         std::vector<std::shared_ptr<Expression>> sizes,
                         Location loc = {}) {
    return std::make_shared<AllocTensor>(elementType, sizes, loc);
}

inline auto dim(std::shared_ptr<Expression> field,
                std::shared_ptr<Expression> index,
                Location loc = {}) {
    return std::make_shared<Dim>(field, index, loc);
}

} // namespace ast