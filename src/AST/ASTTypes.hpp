#pragma once

#include <variant>
#include <cstddef>

namespace types {

struct FundamentalType {
    enum eType {
        SINT8,
        SINT16,
        SINT32,
        SINT64,
        UINT8,
        UINT16,
        UINT32,
        UINT64,
        SSIZE,
        USIZE,
        FLOAT32,
        FLOAT64,
        BOOL,
    };
    FundamentalType() = default;
    FundamentalType(eType type) : type(type) {}
    eType type;
};

struct FieldType {
    FundamentalType elementType;
    size_t numDimensions;
};


using Type = std::variant<FundamentalType, FieldType>;

}