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

inline auto kernel(std::string name,
                   std::vector<Parameter> parameters,
                   std::vector<types::Type> results,
                   std::vector<std::shared_ptr<Statement>> body,
                   size_t numDimensions,
                   std::optional<Location> loc = {}) {
    return std::make_shared<Kernel>(name, parameters, results, body, numDimensions, loc);
}

inline auto kernel_return(std::vector<std::shared_ptr<Expression>> values = {},
                          std::optional<Location> loc = {}) {
    return std::make_shared<KernelReturn>(values, loc);
}

inline auto launch(std::string callee,
                   std::vector<std::shared_ptr<Expression>> gridDim,
                   std::vector<std::shared_ptr<Expression>> arguments = {},
                   std::vector<std::shared_ptr<Expression>> targets = {},
                   std::optional<Location> loc = {}) {
    return std::make_shared<Launch>(callee, gridDim, arguments, targets, loc);
}

inline auto module_(std::vector<std::shared_ptr<Node>> body,
                    std::vector<std::shared_ptr<Kernel>> kernels = {},
                    std::vector<Parameter> parameters = {},
                    std::optional<Location> loc = {}) {
    return std::make_shared<Module>(body, kernels, parameters, loc);
}


//------------------------------------------------------------------------------
// Kernel intrinsics
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

} // namespace ast