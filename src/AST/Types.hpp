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

template <class Visitor>
decltype(auto) VisitType(ScalarType type, Visitor visitor) {
    switch (type) {
        case ScalarType::SINT8: return visitor(static_cast<int8_t*>(nullptr));
        case ScalarType::SINT16: return visitor(static_cast<int16_t*>(nullptr));
        case ScalarType::SINT32: return visitor(static_cast<int32_t*>(nullptr));
        case ScalarType::SINT64: return visitor(static_cast<int64_t*>(nullptr));
        case ScalarType::UINT8: return visitor(static_cast<uint8_t*>(nullptr));
        case ScalarType::UINT16: return visitor(static_cast<uint16_t*>(nullptr));
        case ScalarType::UINT32: return visitor(static_cast<uint32_t*>(nullptr));
        case ScalarType::UINT64: return visitor(static_cast<uint64_t*>(nullptr));
        case ScalarType::INDEX: return visitor(static_cast<ptrdiff_t*>(nullptr));
        case ScalarType::FLOAT32: return visitor(static_cast<float*>(nullptr));
        case ScalarType::FLOAT64: return visitor(static_cast<double*>(nullptr));
        case ScalarType::BOOL: return visitor(static_cast<bool*>(nullptr));
    }
}

} // namespace ast