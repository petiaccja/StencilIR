#include "Invoke.hpp"

#include <optional>


//------------------------------------------------------------------------------
// Utilities
//------------------------------------------------------------------------------

ast::TypePtr GetTypeFromFormat(std::string_view format) {
    using namespace std::string_literals;

    const auto pybindType = [&]() -> ast::TypePtr {
        switch (format[0]) {
            case '?': return std::make_shared<ast::IntegerType>(1, false);
            case 'b': return std::make_shared<ast::IntegerType>(8, true);
            case 'B': return std::make_shared<ast::IntegerType>(8, false);
            case 'h': return std::make_shared<ast::IntegerType>(16, true);
            case 'H': return std::make_shared<ast::IntegerType>(16, false);
            case 'i': return std::make_shared<ast::IntegerType>(32, true);
            case 'I': return std::make_shared<ast::IntegerType>(32, false);
            case 'q': return std::make_shared<ast::IntegerType>(64, true);
            case 'Q': return std::make_shared<ast::IntegerType>(64, false);
            case 'f': return std::make_shared<ast::FloatType>(32);
            case 'd': return std::make_shared<ast::FloatType>(64);
            case 'g': return sizeof(double) == sizeof(long double)
                                 ? std::make_shared<ast::FloatType>(64)
                                 : throw std::invalid_argument("long double is not supported");
        }
        return {};
    }();

    const auto pythonType = [&]() -> ast::TypePtr {
        switch (format[0]) {
            case '?': return ast::InferType<bool>();
            case 'b': return ast::InferType<signed char>();
            case 'B': return ast::InferType<unsigned char>();
            case 'h': return ast::InferType<short int>();
            case 'H': return ast::InferType<unsigned short int>();
            case 'i': return ast::InferType<int>();
            case 'I': return ast::InferType<unsigned int>();
            case 'l': return ast::InferType<long int>();
            case 'k': return ast::InferType<unsigned long int>();
            case 'L': return ast::InferType<long long>();
            case 'K': return ast::InferType<unsigned long long>();
            case 'n': return ast::InferType<Py_ssize_t>();
            case 'f': return ast::InferType<float>();
            case 'd': return ast::InferType<double>();
            case 'p': return ast::InferType<int>();
        }
        return {};
    }();

    // Naturally, they couldn't get it right, so pybind and Python formats
    // not only use different strings for the same thing, but the same strings
    // can also mean different things... Unless both match, we can't know
    // where the data came from.
    if (pybindType && pythonType) {
        const auto pybindIntType = std::dynamic_pointer_cast<ast::IntegerType>(pybindType);
        const auto pythonIntType = std::dynamic_pointer_cast<ast::IntegerType>(pybindType);

        const auto pybindFloatType = std::dynamic_pointer_cast<ast::FloatType>(pybindType);
        const auto pythonFloatType = std::dynamic_pointer_cast<ast::FloatType>(pybindType);

        if (pybindIntType && pythonIntType) {
            if (pybindIntType->size == pythonIntType->size
                && pybindIntType->isSigned == pythonIntType->isSigned) {
                return pybindType;
            }
        }
        if (pybindFloatType && pythonFloatType) {
            if (pybindFloatType->size == pythonFloatType->size) {
                return pybindType;
            }
        }
        std::stringstream ss;
        ss << "ambiguous python format descriptor \"" << format << "\": pybind11"
           << " (" << pybindType << ") "
           << "and python"
           << " (" << pythonType << ") "
           << " interpretations are different";
        throw std::logic_error(ss.str());
    }
    else if (pybindType) {
        return pybindType;
    }
    else if (pythonType) {
        return pythonType;
    }
    throw std::invalid_argument("unsupported python format string: "s + format[0]);
}


static llvm::Type* ConvertType(const ast::Type& type, llvm::LLVMContext& context) {
    if (auto integerType = dynamic_cast<const ast::IntegerType*>(&type)) {
        if (!integerType->isSigned) {
            throw std::invalid_argument("unsigned types are not supported due to arith.constant behavior; TODO: add support");
        }
        return llvm::IntegerType::get(context, integerType->size);
    }
    else if (auto floatType = dynamic_cast<const ast::FloatType*>(&type)) {
        switch (floatType->size) {
            case 32: return llvm::Type::getFloatTy(context);
            case 64: return llvm::Type::getDoubleTy(context);
        }
        throw std::invalid_argument("only 32 and 64-bit floats are supported");
    }
    else if (auto indexType = dynamic_cast<const ast::IndexType*>(&type)) {
        return llvm::IntegerType::get(context, 8 * sizeof(size_t));
    }
    else if (auto fieldType = dynamic_cast<const ast::FieldType*>(&type)) {
        auto elementType = ConvertType(*fieldType->elementType, context);
        auto ptrType = llvm::PointerType::get(elementType, 0);
        auto indexType = llvm::IntegerType::get(context, 8 * sizeof(size_t));
        auto indexArrayType = llvm::ArrayType::get(indexType, fieldType->numDimensions);
        std::array<llvm::Type*, 5> structElements = {
            ptrType,
            ptrType,
            indexType,
            indexArrayType,
            indexArrayType,
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
Argument::Argument(ast::TypePtr type, const Runner* runner)
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

pybind11::object Argument::Read(const void* address) {
    if (auto type = dynamic_cast<const ast::IntegerType*>(m_type.get())) {
        return Read(*type, address);
    }
    else if (auto type = dynamic_cast<const ast::FloatType*>(m_type.get())) {
        return Read(*type, address);
    }
    else if (auto type = dynamic_cast<const ast::IndexType*>(m_type.get())) {
        return Read(*type, address);
    }
    else if (auto type = dynamic_cast<const ast::FieldType*>(m_type.get())) {
        return Read(*type, address);
    }
    std::terminate();
}

void Argument::Write(pybind11::object value, void* address) {
    if (auto type = dynamic_cast<const ast::IntegerType*>(m_type.get())) {
        return Write(*type, value, address);
    }
    else if (auto type = dynamic_cast<const ast::FloatType*>(m_type.get())) {
        return Write(*type, value, address);
    }
    else if (auto type = dynamic_cast<const ast::IndexType*>(m_type.get())) {
        return Write(*type, value, address);
    }
    else if (auto type = dynamic_cast<const ast::FieldType*>(m_type.get())) {
        return Write(*type, value, address);
    }
    std::terminate();
}

//--------------------------------------
// ScalarType methods
//--------------------------------------

pybind11::object Argument::Read(const ast::IntegerType& type, const void* address) const {
    return ast::VisitType(type, [&](auto* typed) -> pybind11::object {
        using T = std::decay_t<decltype(*typed)>;
        const T value = *static_cast<const T*>(address);
        if constexpr (std::is_integral_v<T>) {
            return pybind11::int_(value);
        }
        assert(false && "should only ever receive integers");
        std::terminate();
    });
}

void Argument::Write(const ast::IntegerType& type, pybind11::object value, void* address) const {
    ast::VisitType(type, [&](auto* typed) {
        using T = std::decay_t<decltype(*typed)>;
        *static_cast<T*>(address) = value.cast<T>();
    });
}

pybind11::object Argument::Read(const ast::FloatType& type, const void* address) const {
    return ast::VisitType(type, [&](auto* typed) -> pybind11::object {
        using T = std::decay_t<decltype(*typed)>;
        const T value = *static_cast<const T*>(address);
        if constexpr (std::is_floating_point_v<T>) {
            return pybind11::float_(value);
        }
        assert(false && "should only ever receive floats");
        std::terminate();
    });
}

void Argument::Write(const ast::FloatType& type, pybind11::object value, void* address) const {
    ast::VisitType(type, [&](auto* typed) {
        using T = std::decay_t<decltype(*typed)>;
        *static_cast<T*>(address) = value.cast<T>();
    });
}

pybind11::object Argument::Read(const ast::IndexType& type, const void* address) const {
    return ast::VisitType(type, [&](auto* typed) -> pybind11::object {
        using T = std::decay_t<decltype(*typed)>;
        const T value = *static_cast<const T*>(address);
        if constexpr (std::is_integral_v<T>) {
            return pybind11::int_(value);
        }
        assert(false && "should only ever receive integers");
        std::terminate();
    });
}

void Argument::Write(const ast::IndexType& type, pybind11::object value, void* address) const {
    ast::VisitType(type, [&](auto* typed) {
        using T = std::decay_t<decltype(*typed)>;
        *static_cast<T*>(address) = value.cast<T>();
    });
}

//--------------------------------------
// Field type methods
//--------------------------------------
pybind11::object Argument::Read(const ast::FieldType& type, const void* address) const {
    auto layout = GetLayout();
    assert(layout);

    return ast::VisitType(*type.elementType, [&](auto* typed) {
        using T = std::decay_t<decltype(*typed)>;

        const auto startingAddress = reinterpret_cast<const std::byte*>(address);
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

void Argument::Write(const ast::FieldType& type, pybind11::object value, void* address) const {
    auto layout = GetLayout();
    assert(layout);

    const auto elementSize = m_runner->GetDataLayout().getTypeSizeInBits(ConvertType(*type.elementType, m_runner->GetContext())) / 8;

    const auto buffer = value.cast<pybind11::buffer>();
    const auto bufferInfo = buffer.request(true);
    const auto bufferType = ast::FieldType{ GetTypeFromFormat(bufferInfo.format), int(bufferInfo.ndim) };
    if (!bufferType.EqualTo(type)) {
        std::stringstream ss;
        ss << "expected buffer type " << type << ", got " << bufferType;
        throw std::invalid_argument(ss.str());
    }

    const auto startingAddress = reinterpret_cast<std::byte*>(address);
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
ArgumentPack::ArgumentPack(std::span<const ast::TypePtr> types, const Runner* runner)
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

pybind11::object ArgumentPack::Read(const void* address) {
    const auto numItems = m_items.size();
    const auto startingAddress = reinterpret_cast<const std::byte*>(address);
    auto values = pybind11::tuple{ numItems };
    for (size_t i = 0; i < numItems; ++i) {
        const auto offset = GetLayout()->getElementOffset(i);
        values[i] = m_items[i].Read(startingAddress + offset);
    }
    return values;
}

void ArgumentPack::Write(pybind11::object value, void* address) {
    const auto numItems = m_items.size();
    const auto valuesAsTuple = value.cast<pybind11::tuple>();
    const auto numSupplied = valuesAsTuple.size();
    if (numItems != numSupplied) {
        std::stringstream ss;
        ss << "expected " << numItems << " values, got " << numSupplied;
        throw std::invalid_argument(ss.str());
    }

    const auto startingAddress = reinterpret_cast<std::byte*>(address);
    auto values = pybind11::tuple{ numItems };
    for (size_t i = 0; i < numItems; ++i) {
        const auto offset = GetLayout()->getElementOffset(i);
        m_items[i].Write(valuesAsTuple[i], startingAddress + offset);
    }
}

const llvm::StructLayout* ArgumentPack::GetLayout() const {
    return m_runner->GetDataLayout().getStructLayout(m_llvmType);
}
