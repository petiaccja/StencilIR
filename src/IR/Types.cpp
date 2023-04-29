#include "Types.hpp"


namespace sir {


Type::~Type() = default;


//------------------------------------------------------------------------------
// IntegerType
//------------------------------------------------------------------------------

IntegerType::IntegerType(int size, bool isSigned) : size(size), isSigned(isSigned) {
    if (size != 1 && size != 8 && size != 16 && size != 32 && size != 64) {
        throw std::invalid_argument("integer type must be 1, 8, 16, 32, or 64-bit");
    }
}

bool IntegerType::EqualTo(const Type& other) const {
    if (auto otherInt = dynamic_cast<const IntegerType*>(&other)) {
        return size == otherInt->size && isSigned == otherInt->isSigned;
    }
    return false;
}

std::ostream& IntegerType::Print(std::ostream& os) const {
    return os << (isSigned ? 's' : 'u') << 'i' << size;
}

std::shared_ptr<IntegerType> IntegerType::Get(int size, bool isSigned) {
    return std::make_shared<IntegerType>(size, isSigned);
}


//------------------------------------------------------------------------------
// FloatType
//------------------------------------------------------------------------------

FloatType::FloatType(int size) : size(size) {
    if (size != 16 && size != 32 && size != 64 && size != 128) {
        throw std::invalid_argument("float type must be 16, 32, 64 or 128-bit");
    }
}

bool FloatType::EqualTo(const Type& other) const {
    if (auto otherFloat = dynamic_cast<const FloatType*>(&other)) {
        return size == otherFloat->size;
    }
    return false;
}

std::ostream& FloatType::Print(std::ostream& os) const {
    return os << 'f' << size;
}

std::shared_ptr<FloatType> FloatType::Get(int size) {
    return std::make_shared<FloatType>(size);
}


//------------------------------------------------------------------------------
// IndexType
//------------------------------------------------------------------------------

bool IndexType::EqualTo(const Type& other) const {
    if (auto otherIndex = dynamic_cast<const IndexType*>(&other)) {
        return true;
    }
    return false;
}

std::ostream& IndexType::Print(std::ostream& os) const {
    return os << "index";
}

std::shared_ptr<IndexType> IndexType::Get() {
    return std::make_shared<IndexType>();
}


//------------------------------------------------------------------------------
// NDIndexType
//------------------------------------------------------------------------------

NDIndexType::NDIndexType(int numDimensions) : numDimensions(numDimensions) {}

bool NDIndexType::EqualTo(const Type& other) const {
    if (auto otherNDIndex = dynamic_cast<const NDIndexType*>(&other)) {
        return numDimensions == otherNDIndex->numDimensions;
    }
    return false;
}

std::ostream& NDIndexType::Print(std::ostream& os) const {
    return os << "index<" << numDimensions << ">";
}

std::shared_ptr<NDIndexType> NDIndexType::Get(int numDimensions) {
    return std::make_shared<NDIndexType>(numDimensions);
}


//------------------------------------------------------------------------------
// FieldType
//------------------------------------------------------------------------------


FieldType::FieldType(std::shared_ptr<Type> elementType, int numDimensions)
    : elementType(elementType),
      numDimensions(numDimensions) {}

bool FieldType::EqualTo(const Type& other) const {
    if (auto otherField = dynamic_cast<const FieldType*>(&other)) {
        return elementType->EqualTo(*otherField->elementType) && numDimensions == otherField->numDimensions;
    }
    return false;
}

std::ostream& FieldType::Print(std::ostream& os) const {
    os << "field<" << *elementType;
    for (int i = 0; i < numDimensions; ++i) {
        os << "x?";
    }
    os << ">";
    return os;
}

std::shared_ptr<FieldType> FieldType::Get(std::shared_ptr<Type> elementType, int numDimensions) {
    return std::make_shared<FieldType>(elementType, numDimensions);
}


//------------------------------------------------------------------------------
// FunctionType
//------------------------------------------------------------------------------

FunctionType::FunctionType(std::vector<std::shared_ptr<Type>> parameters,
                           std::vector<std::shared_ptr<Type>> results)
    : parameters(std::move(parameters)), results(std::move(results)) {}

bool FunctionType::EqualTo(const Type& other) const {
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

std::ostream& FunctionType::Print(std::ostream& os) const {
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

std::shared_ptr<FunctionType> FunctionType::Get(std::vector<std::shared_ptr<Type>> parameters,
                                                std::vector<std::shared_ptr<Type>> results) {
    return std::make_shared<FunctionType>(std::move(parameters), std::move(results));
}


} // namespace sir