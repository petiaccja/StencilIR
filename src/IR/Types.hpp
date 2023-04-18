#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <sstream>
#include <vector>


namespace sir {


class Type {
public:
    virtual ~Type() = default;
    virtual bool EqualTo(const Type& other) const = 0;
    virtual std::ostream& Print(std::ostream& os) const = 0;

    friend std::ostream& operator<<(std::ostream& os, const Type& type) {
        return type.Print(os);
    }
};


class IntegerType : public Type {
public:
    IntegerType(int size, bool isSigned) : size(size), isSigned(isSigned) {
        if (size != 1 && size != 8 && size != 16 && size != 32 && size != 64) {
            throw std::invalid_argument("integer type must be 1, 8, 16, 32, or 64-bit");
        }
    }

    bool EqualTo(const Type& other) const override {
        if (auto otherInt = dynamic_cast<const IntegerType*>(&other)) {
            return size == otherInt->size && isSigned == otherInt->isSigned;
        }
        return false;
    }

    std::ostream& Print(std::ostream& os) const override {
        return os << (isSigned ? 's' : 'u') << 'i' << size;
    }

    static auto Get(int size, bool isSigned) {
        return std::make_shared<IntegerType>(size, isSigned);
    }

public:
    const int size;
    const bool isSigned;
};


class FloatType : public Type {
public:
    explicit FloatType(int size) : size(size) {
        if (size != 16 && size != 32 && size != 64) {
            throw std::invalid_argument("float type must be 16, 32, or 64-bit");
        }
    }

    bool EqualTo(const Type& other) const override {
        if (auto otherFloat = dynamic_cast<const FloatType*>(&other)) {
            return size == otherFloat->size;
        }
        return false;
    }

    std::ostream& Print(std::ostream& os) const override {
        return os << 'f' << size;
    }

    static auto Get(int size) {
        return std::make_shared<FloatType>(size);
    }

public:
    const int size;
};


class IndexType : public Type {
public:
    bool EqualTo(const Type& other) const override {
        if (auto otherIndex = dynamic_cast<const IndexType*>(&other)) {
            return true;
        }
        return false;
    }

    std::ostream& Print(std::ostream& os) const override {
        return os << "index";
    }

    static auto Get() {
        return std::make_shared<IndexType>();
    }
};


class FieldType : public Type {
public:
    FieldType(std::shared_ptr<Type> elementType, int numDimensions)
        : elementType(elementType),
          numDimensions(numDimensions) {}

    bool EqualTo(const Type& other) const override {
        if (auto otherField = dynamic_cast<const FieldType*>(&other)) {
            return elementType->EqualTo(*otherField->elementType) && numDimensions == otherField->numDimensions;
        }
        return false;
    }

    std::ostream& Print(std::ostream& os) const override {
        os << "field<" << *elementType;
        for (int i = 0; i < numDimensions; ++i) {
            os << "x?";
        }
        os << ">";
        return os;
    }

    static auto Get(std::shared_ptr<Type> elementType, int numDimensions) {
        return std::make_shared<FieldType>(elementType, numDimensions);
    }

public:
    const std::shared_ptr<Type> elementType;
    const int numDimensions;
};


class FunctionType : public Type {
public:
    FunctionType(std::vector<std::shared_ptr<Type>> parameters,
                 std::vector<std::shared_ptr<Type>> results)
        : parameters(std::move(parameters)), results(std::move(results)) {}

    bool EqualTo(const Type& other) const override {
        if (auto otherFunction = dynamic_cast<const FunctionType*>(&other)) {
            if (parameters.size() != otherFunction->parameters.size()) {
                return false;
            }
            for (auto [itl, itr] = std::tuple{ parameters.begin(), otherFunction->parameters.begin() };
                 itl != parameters.end();
                 ++itl, ++itr) {
                if (!(*itl)->EqualTo(**itr)) {
                    return false;
                }
            }
            if (results.size() != otherFunction->results.size()) {
                return false;
            }
            for (auto [itl, itr] = std::tuple{ results.begin(), otherFunction->results.begin() };
                 itl != results.end();
                 ++itl, ++itr) {
                if (!(*itl)->EqualTo(**itr)) {
                    return false;
                }
            }
            return true;
        }
        return false;
    }

    std::ostream& Print(std::ostream& os) const override {
        os << "(";
        for (auto p : parameters) {
            os << p << (p != parameters.back() ? ", " : "");
        }
        os << ") -> ";
        for (auto p : results) {
            os << p << (p != results.back() ? ", " : "");
        }
        return os;
    }

    static auto Get(std::vector<std::shared_ptr<Type>> parameters,
                    std::vector<std::shared_ptr<Type>> results) {
        return std::make_shared<FunctionType>(std::move(parameters), std::move(results));
    }

public:
    std::vector<std::shared_ptr<Type>> parameters;
    std::vector<std::shared_ptr<Type>> results;
};


using TypePtr = std::shared_ptr<Type>;


template <class T>
TypePtr InferType() {
    if constexpr (std::is_integral_v<T>) {
        const int size = std::is_same_v<T, bool> ? 1 : int(8 * sizeof(T));
        const bool isSigned = std::is_same_v<T, bool> ? true : std::is_signed_v<T>;
        return std::make_shared<IntegerType>(size, isSigned);
    }
    else if constexpr (std::is_floating_point_v<T>) {
        const int size = int(8 * sizeof(T));
        return std::make_shared<FloatType>(size);
    }
    else {
        static_assert(sizeof(T*) == 0, "cannot convert C++ type to AST type");
    }
}


template <class Visitor>
decltype(auto) VisitType(const Type& type, Visitor&& visitor) {
    if (auto type_ = dynamic_cast<const IntegerType*>(&type)) {
        switch (type_->size) {
            case 1: return visitor(static_cast<bool*>(nullptr));
            case 8: return type_->isSigned ? visitor(static_cast<int8_t*>(nullptr)) : visitor(static_cast<uint8_t*>(nullptr));
            case 16: return type_->isSigned ? visitor(static_cast<int16_t*>(nullptr)) : visitor(static_cast<uint16_t*>(nullptr));
            case 32: return type_->isSigned ? visitor(static_cast<int32_t*>(nullptr)) : visitor(static_cast<uint32_t*>(nullptr));
            case 64: return type_->isSigned ? visitor(static_cast<int64_t*>(nullptr)) : visitor(static_cast<uint64_t*>(nullptr));
        }
    }
    else if (auto type_ = dynamic_cast<const FloatType*>(&type)) {
        switch (type_->size) {
            case 32: return visitor(static_cast<float*>(nullptr));
            case 64: return visitor(static_cast<double*>(nullptr));
        }
    }
    else if (auto type_ = dynamic_cast<const IndexType*>(&type)) {
        return visitor(static_cast<ptrdiff_t*>(nullptr));
    }
    std::stringstream ss;
    ss << "type \"" << type << "\"cannot be translated to a simple C++ type";
    throw std::invalid_argument(ss.str());
}


inline const auto Float32 = FloatType::Get(32);
inline const auto Float64 = FloatType::Get(64);

inline const auto Int8 = IntegerType::Get(8, true);
inline const auto Int16 = IntegerType::Get(16, true);
inline const auto Int32 = IntegerType::Get(32, true);
inline const auto Int64 = IntegerType::Get(64, true);

inline const auto Uint8 = IntegerType::Get(8, false);
inline const auto Uint16 = IntegerType::Get(16, false);
inline const auto Uint32 = IntegerType::Get(32, false);
inline const auto Uint64 = IntegerType::Get(64, false);

inline const auto Bool = IntegerType::Get(1, true);


} // namespace sir