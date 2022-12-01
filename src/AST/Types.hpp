#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <ostream>
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

    bool operator==(const FieldType&) const = default;
    bool operator!=(const FieldType&) const = default;
    bool operator<=>(const FieldType&) const = default;
};

using Type = std::variant<ScalarType, FieldType>;

template <class T>
ScalarType InferType() {
    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
        if constexpr (std::is_signed_v<T>) {
            if constexpr (sizeof(T) == 1) {
                return ScalarType::SINT8;
            }
            else if constexpr (sizeof(T) == 2) {
                return ScalarType::SINT16;
            }
            else if constexpr (sizeof(T) == 4) {
                return ScalarType::SINT32;
            }
            else if constexpr (sizeof(T) == 8) {
                return ScalarType::SINT64;
            }
            else {
                static_assert(!sizeof(T*), "signed int type not supported in the AST type system");
            }
        }
        else {
            if constexpr (sizeof(T) == 1) {
                return ScalarType::UINT8;
            }
            else if constexpr (sizeof(T) == 2) {
                return ScalarType::UINT16;
            }
            else if constexpr (sizeof(T) == 4) {
                return ScalarType::UINT32;
            }
            else if constexpr (sizeof(T) == 8) {
                return ScalarType::UINT64;
            }
            else {
                static_assert(!sizeof(T*), "unsigned int type not supported in the AST type system");
            }
        }
    }
    else if constexpr (std::is_same_v<T, float>) {
        return ScalarType::FLOAT32;
    }
    else if constexpr (std::is_same_v<T, double>) {
        return ScalarType::FLOAT64;
    }
    else if constexpr (std::is_same_v<T, bool>) {
        return ScalarType::BOOL;
    }
    else {
        static_assert(!sizeof(T*), "C++ type not supported in the AST type system");
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


inline std::ostream& operator<<(std::ostream& os, const ScalarType& obj) {
    switch (obj) {
        case ScalarType::SINT8: return os << "si8";
        case ScalarType::SINT16: return os << "si16";
        case ScalarType::SINT32: return os << "si32";
        case ScalarType::SINT64: return os << "si64";
        case ScalarType::UINT8: return os << "ui8";
        case ScalarType::UINT16: return os << "ui16";
        case ScalarType::UINT32: return os << "ui32";
        case ScalarType::UINT64: return os << "ui64";
        case ScalarType::INDEX: return os << "index";
        case ScalarType::FLOAT32: return os << "f32";
        case ScalarType::FLOAT64: return os << "f64";
        case ScalarType::BOOL: return os << "i1";
    }
    return os;
}


inline std::ostream& operator<<(std::ostream& os, const FieldType& obj) {
    os << "field<";
    for (size_t i = 0; i < obj.numDimensions; ++i) {
        os << "?x";
    }
    os << obj.elementType << ">";
    return os;
}


inline std::ostream& operator<<(std::ostream& os, const Type& obj) {
    std::visit([&](auto obj) { os << obj; }, obj);
    return os;
}


} // namespace ast