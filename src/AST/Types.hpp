#pragma once

#include <cstddef>
#include <cstdint>
#include <variant>

namespace ast {

namespace impl {
    struct IndexType {};
} // namespace impl

constexpr impl::IndexType index_type;

enum class ScalarType {
    SINT8,
    SINT16,
    SINT32,
    SINT64,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    INDEX,
    FLOAT32,
    FLOAT64,
    BOOL,
};

struct FieldType {
    ScalarType elementType;
    size_t numDimensions;
};

using Type = std::variant<ScalarType, FieldType>;

template <class T>
ScalarType InferType() {
    if constexpr (std::is_same_v<T, int8_t>) {
        return ScalarType::SINT8;
    }
    if constexpr (std::is_same_v<T, int16_t>) {
        return ScalarType::SINT16;
    }
    if constexpr (std::is_same_v<T, int32_t>) {
        return ScalarType::SINT32;
    }
    if constexpr (std::is_same_v<T, int64_t>) {
        return ScalarType::SINT64;
    }
    if constexpr (std::is_same_v<T, uint8_t>) {
        return ScalarType::UINT8;
    }
    if constexpr (std::is_same_v<T, uint16_t>) {
        return ScalarType::UINT16;
    }
    if constexpr (std::is_same_v<T, uint32_t>) {
        return ScalarType::UINT32;
    }
    if constexpr (std::is_same_v<T, uint64_t>) {
        return ScalarType::UINT64;
    }
    if constexpr (std::is_same_v<T, float>) {
        return ScalarType::FLOAT32;
    }
    if constexpr (std::is_same_v<T, double>) {
        return ScalarType::FLOAT64;
    }
    if constexpr (std::is_same_v<T, bool>) {
        return ScalarType::BOOL;
    }
}

} // namespace types