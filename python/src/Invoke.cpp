#include "Invoke.hpp"

#include <optional>


namespace sir {

//------------------------------------------------------------------------------
// Utilities
//------------------------------------------------------------------------------

TypePtr GetTypeFromFormat(std::string_view format) {
    using namespace std::string_literals;

    const auto pybindType = [&]() -> TypePtr {
        switch (format[0]) {
            case '?': return std::make_shared<IntegerType>(1, true);
            case 'b': return std::make_shared<IntegerType>(8, true);
            case 'B': return std::make_shared<IntegerType>(8, false);
            case 'h': return std::make_shared<IntegerType>(16, true);
            case 'H': return std::make_shared<IntegerType>(16, false);
            case 'i': return std::make_shared<IntegerType>(32, true);
            case 'I': return std::make_shared<IntegerType>(32, false);
            case 'q': return std::make_shared<IntegerType>(64, true);
            case 'Q': return std::make_shared<IntegerType>(64, false);
            case 'e': return std::make_shared<FloatType>(16);
            case 'f': return std::make_shared<FloatType>(32);
            case 'd': return std::make_shared<FloatType>(64);
            case 'g': return sizeof(long double) == 8    ? std::make_shared<FloatType>(64)
                             : sizeof(long double) == 16 ? std::make_shared<FloatType>(128)
                                                         : throw std::invalid_argument("long double is not supported");
        }
        return {};
    }();

    const auto pythonType = [&]() -> TypePtr {
        switch (format[0]) {
            case '?': return InferType<bool>();
            case 'b': return InferType<signed char>();
            case 'B': return InferType<unsigned char>();
            case 'h': return InferType<short int>();
            case 'H': return InferType<unsigned short int>();
            case 'i': return InferType<int>();
            case 'I': return InferType<unsigned int>();
            case 'l': return InferType<long int>();
            case 'k': return InferType<unsigned long int>();
            case 'L': return InferType<long long>();
            case 'K': return InferType<unsigned long long>();
            case 'n': return InferType<Py_ssize_t>();
            case 'e': return std::make_shared<FloatType>(16);
            case 'f': return InferType<float>();
            case 'd': return InferType<double>();
            case 'g': return sizeof(long double) == 8    ? std::make_shared<FloatType>(64)
                             : sizeof(long double) == 16 ? std::make_shared<FloatType>(128)
                                                         : throw std::invalid_argument("long double is not supported");
            case 'p': return InferType<int>();
        }
        return {};
    }();

    // Naturally, they couldn't get it right, so pybind and Python formats
    // not only use different strings for the same thing, but the same strings
    // can also mean different things... Unless both match, we can't know
    // where the data came from.
    if (pybindType && pythonType) {
        const auto pybindIntType = std::dynamic_pointer_cast<IntegerType>(pybindType);
        const auto pythonIntType = std::dynamic_pointer_cast<IntegerType>(pybindType);

        const auto pybindFloatType = std::dynamic_pointer_cast<FloatType>(pybindType);
        const auto pythonFloatType = std::dynamic_pointer_cast<FloatType>(pybindType);

        if (pybindIntType && pythonIntType
            && pybindIntType->size == pythonIntType->size
            && pybindIntType->isSigned == pythonIntType->isSigned) {
            return pybindType;
        }
        if (pybindFloatType && pythonFloatType
            && pybindFloatType->size == pythonFloatType->size) {
            return pybindType;
        }
        std::stringstream ss;
        ss << "ambiguous python format descriptor \"" << format << "\": pybind11"
           << " (" << pybindType << ") "
           << "and python"
           << " (" << pythonType << ") "
           << " interpretations are different";
        throw std::runtime_error(ss.str());
    }
    else if (pybindType) {
        return pybindType;
    }
    else if (pythonType) {
        return pythonType;
    }
    throw std::invalid_argument("unsupported python format string: "s + format[0]);
}


static llvm::Type* ConvertType(const Type& type, llvm::LLVMContext& context) {
    if (auto integerType = dynamic_cast<const IntegerType*>(&type)) {
        if (!integerType->isSigned) {
            throw std::invalid_argument("unsigned types are not supported due to arith.constant behavior; TODO: add support");
        }
        return llvm::IntegerType::get(context, integerType->size);
    }
    else if (auto floatType = dynamic_cast<const FloatType*>(&type)) {
        switch (floatType->size) {
            case 16: return llvm::Type::getHalfTy(context);
            case 32: return llvm::Type::getFloatTy(context);
            case 64: return llvm::Type::getDoubleTy(context);
            case 128: return llvm::Type::getFP128Ty(context);
        }
        throw std::invalid_argument("only 16, 32, 64, and 128-bit floats are supported");
    }
    else if (auto indexType = dynamic_cast<const IndexType*>(&type)) {
        return llvm::IntegerType::get(context, 8 * sizeof(size_t));
    }
    else if (auto fieldType = dynamic_cast<const FieldType*>(&type)) {
        auto elementType = ConvertType(*fieldType->elementType, context);
        auto ptrType = llvm::PointerType::get(elementType, 0);
        auto llvmIndexType = llvm::IntegerType::get(context, 8 * sizeof(size_t));
        auto llvmIndexArrayType = llvm::ArrayType::get(llvmIndexType, fieldType->numDimensions);
        std::array<llvm::Type*, 5> structElements = {
            ptrType,
            ptrType,
            llvmIndexType,
            llvmIndexArrayType,
            llvmIndexArrayType,
        };
        auto structType = llvm::StructType::get(context, structElements, false);
        return structType;
    }
    else {
        std::stringstream ss;
        ss << "could not convert type \"" << type << "\" to LLVM type";
        throw std::invalid_argument(ss.str());
    }
}

//------------------------------------------------------------------------------
// Argument
//------------------------------------------------------------------------------
Argument::Argument(TypePtr type, const Runner* runner)
    : m_type(type),
      m_runner(runner),
      m_llvmType(ConvertType(*type, runner->GetContext())) {
}

//--------------------------------------
// Generic methods
//--------------------------------------

size_t Argument::GetSize() const {
    return m_runner->GetDataLayout().getTypeSizeInBits(m_llvmType) / 8;
}

size_t Argument::GetAlignment() const {
    if (auto layout = GetLayout()) {
        return layout->getAlignment().value();
    }
    return GetSize();
}

pybind11::object Argument::Read(const std::byte* address) const {
    if (auto type = dynamic_cast<const IntegerType*>(m_type.get())) {
        return Read(*type, address);
    }
    else if (auto type = dynamic_cast<const FloatType*>(m_type.get())) {
        return Read(*type, address);
    }
    else if (auto type = dynamic_cast<const IndexType*>(m_type.get())) {
        return Read(*type, address);
    }
    else if (auto type = dynamic_cast<const FieldType*>(m_type.get())) {
        return Read(*type, address);
    }
    std::terminate();
}

void Argument::Write(pybind11::object value, std::byte* address) const {
    if (auto type = dynamic_cast<const IntegerType*>(m_type.get())) {
        return Write(*type, value, address);
    }
    else if (auto type = dynamic_cast<const FloatType*>(m_type.get())) {
        return Write(*type, value, address);
    }
    else if (auto type = dynamic_cast<const IndexType*>(m_type.get())) {
        return Write(*type, value, address);
    }
    else if (auto type = dynamic_cast<const FieldType*>(m_type.get())) {
        return Write(*type, value, address);
    }
    std::terminate();
}

//--------------------------------------
// ScalarType methods
//--------------------------------------

pybind11::object Argument::Read(const IntegerType& type, const std::byte* address) const {
    return VisitType(type, [&](auto* typed) -> pybind11::object {
        using T = std::decay_t<decltype(*typed)>;
        const T value = *reinterpret_cast<const T*>(address);
        if constexpr (std::is_integral_v<T>) {
            return pybind11::int_(value);
        }
        assert(false && "should only ever receive integers");
        std::terminate();
    });
}

void Argument::Write(const IntegerType& type, pybind11::object value, std::byte* address) const {
    VisitType(type, [&](auto* typed) {
        using T = std::decay_t<decltype(*typed)>;
        *reinterpret_cast<T*>(address) = value.cast<T>();
    });
}

pybind11::object Argument::Read(const FloatType& type, const std::byte* address) const {
    return VisitType(type, [&](auto* typed) -> pybind11::object {
        using T = std::decay_t<decltype(*typed)>;
        const T value = *reinterpret_cast<const T*>(address);
        if constexpr (std::is_floating_point_v<T>) {
            return pybind11::float_(value);
        }
        assert(false && "should only ever receive floats");
        std::terminate();
    });
}

void Argument::Write(const FloatType& type, pybind11::object value, std::byte* address) const {
    VisitType(type, [&](auto* typed) {
        using T = std::decay_t<decltype(*typed)>;
        *reinterpret_cast<T*>(address) = value.cast<T>();
    });
}

pybind11::object Argument::Read(const IndexType& type, const std::byte* address) const {
    return VisitType(type, [&](auto* typed) -> pybind11::object {
        using T = std::decay_t<decltype(*typed)>;
        const T value = *reinterpret_cast<const T*>(address);
        if constexpr (std::is_integral_v<T>) {
            return pybind11::int_(value);
        }
        assert(false && "should only ever receive integers");
        std::terminate();
    });
}

void Argument::Write(const IndexType& type, pybind11::object value, std::byte* address) const {
    VisitType(type, [&](auto* typed) {
        using T = std::decay_t<decltype(*typed)>;
        *reinterpret_cast<T*>(address) = value.cast<T>();
    });
}

//--------------------------------------
// Field type methods
//--------------------------------------
pybind11::object Argument::Read(const FieldType& type, const std::byte* address) const {
    auto layout = GetLayout();
    assert(layout);

    return VisitType(*type.elementType, [&](auto* typed) {
        using T = std::decay_t<decltype(*typed)>;

        const auto startingAddress = address;
        const auto alignedPtrAddress = reinterpret_cast<T* const*>(startingAddress + layout->getElementOffset(1));
        const auto offsetAddress = reinterpret_cast<const ptrdiff_t*>(startingAddress + layout->getElementOffset(2));
        const auto shapeAddress = reinterpret_cast<const ptrdiff_t*>(startingAddress + layout->getElementOffset(3));
        const auto stridesAddress = reinterpret_cast<const ptrdiff_t*>(startingAddress + layout->getElementOffset(4));
        llvm::SmallVector<ptrdiff_t, 12> byteStrides;
        byteStrides.reserve(type.numDimensions);
        std::transform(stridesAddress, stridesAddress + type.numDimensions,
                       std::back_inserter(byteStrides),
                       [&](auto s) { return s * sizeof(T); });
        return pybind11::memoryview::from_buffer(*alignedPtrAddress + *offsetAddress,
                                                 std::span{ shapeAddress, size_t(type.numDimensions) },
                                                 std::span{ byteStrides },
                                                 false);
    });
}

void Argument::Write(const FieldType& type, pybind11::object value, std::byte* address) const {
    auto layout = GetLayout();
    assert(layout);

    const auto llvmType = ConvertType(*type.elementType, m_runner->GetContext());
    const auto elementSize = m_runner->GetDataLayout().getTypeAllocSize(llvmType);

    const auto buffer = value.cast<pybind11::buffer>();
    const auto bufferInfo = buffer.request(true);
    const auto bufferType = FieldType{ GetTypeFromFormat(bufferInfo.format), int(bufferInfo.ndim) };
    if (!bufferType.EqualTo(type)) {
        const auto bufferElementIntType = dynamic_cast<sir::IntegerType*>(bufferType.elementType.get());
        const auto elementIndexType = dynamic_cast<sir::IntegerType*>(type.elementType.get());
        if (bufferElementIntType && elementIndexType && bufferElementIntType->size != elementSize * 8) {
            std::stringstream ss;
            ss << "expected buffer type " << type << ", got " << bufferType;
            throw std::invalid_argument(ss.str());
        }
    }

    const auto startingAddress = address;
    const auto allocPtrAddress = reinterpret_cast<std::byte**>(startingAddress + layout->getElementOffset(0));
    const auto alignedPtrAddress = reinterpret_cast<std::byte**>(startingAddress + layout->getElementOffset(1));
    const auto offsetAddress = reinterpret_cast<ptrdiff_t*>(startingAddress + layout->getElementOffset(2));
    const auto shapeAddress = reinterpret_cast<ptrdiff_t*>(startingAddress + layout->getElementOffset(3));
    const auto stridesAddress = reinterpret_cast<ptrdiff_t*>(startingAddress + layout->getElementOffset(4));

    *allocPtrAddress = reinterpret_cast<std::byte*>(bufferInfo.ptr);
    *alignedPtrAddress = *allocPtrAddress;
    *offsetAddress = 0;
    std::copy(bufferInfo.shape.begin(), bufferInfo.shape.end(), shapeAddress);
    std::transform(bufferInfo.strides.begin(), bufferInfo.strides.end(), stridesAddress, [&](auto s) { return s / elementSize; });
}

const llvm::StructLayout* Argument::GetLayout() const {
    if (auto structType = llvm::dyn_cast<llvm::StructType>(m_llvmType)) {
        return m_runner->GetDataLayout().getStructLayout(structType);
    }
    return nullptr;
}


//------------------------------------------------------------------------------
// ArgumentPack
//------------------------------------------------------------------------------
ArgumentPack::ArgumentPack(std::span<const TypePtr> types, const Runner* runner)
    : m_runner(runner) {
    m_items.reserve(types.size());
    for (auto& type : types) {
        m_items.push_back(Argument(type, runner));
    }

    llvm::SmallVector<llvm::Type*, 16> itemTypes;
    itemTypes.reserve(m_items.size());
    for (auto& item : m_items) {
        itemTypes.push_back(item.GetLLVMType());
    }
    m_llvmType = llvm::StructType::get(runner->GetContext(), itemTypes);
}

size_t ArgumentPack::GetSize() const {
    return GetLayout()->getSizeInBytes();
}

size_t ArgumentPack::GetAlignment() const {
    return GetLayout()->getAlignment().value();
}

pybind11::object ArgumentPack::Read(const std::byte* address) const {
    const auto numItems = m_items.size();
    const auto startingAddress = address;
    auto values = pybind11::tuple{ numItems };
    for (size_t i = 0; i < numItems; ++i) {
        const auto offset = GetLayout()->getElementOffset(i);
        values[i] = m_items[i].Read(startingAddress + offset);
    }
    return values;
}

void ArgumentPack::Write(pybind11::object value, std::byte* address) const {
    const auto numItems = m_items.size();
    const auto valuesAsTuple = value.cast<pybind11::tuple>();
    const auto numSupplied = valuesAsTuple.size();
    if (numItems != numSupplied) {
        std::stringstream ss;
        ss << "expected " << numItems << " values, got " << numSupplied;
        throw std::invalid_argument(ss.str());
    }

    const auto startingAddress = address;
    auto values = pybind11::tuple{ numItems };
    for (size_t i = 0; i < numItems; ++i) {
        const auto offset = GetLayout()->getElementOffset(i);
        m_items[i].Write(valuesAsTuple[i], startingAddress + offset);
    }
}

const llvm::StructLayout* ArgumentPack::GetLayout() const {
    return m_runner->GetDataLayout().getStructLayout(m_llvmType);
}


} // namespace sir