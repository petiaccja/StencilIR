#include "Invoke.hpp"

#include <optional>


//------------------------------------------------------------------------------
// Utilities
//------------------------------------------------------------------------------

ast::ScalarType GetTypeFromFormat(std::string_view format) {
    using namespace std::string_literals;

    const auto pybindType = [&]() -> std::optional<ast::ScalarType> {
        switch (format[0]) {
            case '?': return ast::ScalarType::BOOL;
            case 'b': return ast::ScalarType::SINT8;
            case 'B': return ast::ScalarType::UINT8;
            case 'h': return ast::ScalarType::SINT16;
            case 'H': return ast::ScalarType::UINT16;
            case 'i': return ast::ScalarType::SINT32;
            case 'I': return ast::ScalarType::UINT32;
            case 'q': return ast::ScalarType::SINT64;
            case 'Q': return ast::ScalarType::UINT64;
            case 'f': return ast::ScalarType::FLOAT32;
            case 'd': return ast::ScalarType::FLOAT64;
            case 'g': return sizeof(double) == sizeof(long double)
                                 ? ast::ScalarType::FLOAT64
                                 : throw std::invalid_argument("long double is not supported");
        }
        return {};
    }();

    const auto pythonType = [&]() -> std::optional<ast::ScalarType> {
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
        if (pybindType.value() == pybindType.value()) {
            return pybindType.value();
        }
        throw std::logic_error("ambiguous python format descriptor: pybind11 and python interpretation is different");
    }
    else if (pybindType) {
        return pybindType.value();
    }
    else if (pythonType) {
        return pythonType.value();
    }
    throw std::invalid_argument("unsupported python format string: "s + format[0]);
}


static llvm::Type* ConvertType(ast::Type type, llvm::LLVMContext& context) {
    struct Visitor {
        llvm::Type* operator()(ast::ScalarType type) const {
            switch (type) {
                case ast::ScalarType::SINT8: return llvm::IntegerType::get(context, 8);
                case ast::ScalarType::SINT16: return llvm::IntegerType::get(context, 16);
                case ast::ScalarType::SINT32: return llvm::IntegerType::get(context, 32);
                case ast::ScalarType::SINT64: return llvm::IntegerType::get(context, 64);
                case ast::ScalarType::UINT8: return llvm::IntegerType::get(context, 8);
                case ast::ScalarType::UINT16: return llvm::IntegerType::get(context, 16);
                case ast::ScalarType::UINT32: return llvm::IntegerType::get(context, 32);
                case ast::ScalarType::UINT64: return llvm::IntegerType::get(context, 64);
                case ast::ScalarType::INDEX: return llvm::IntegerType::get(context, 8 * sizeof(size_t));
                case ast::ScalarType::FLOAT32: return llvm::Type::getFloatTy(context);
                case ast::ScalarType::FLOAT64: return llvm::Type::getDoubleTy(context);
                case ast::ScalarType::BOOL: return llvm::Type::getInt1Ty(context);
            }
            throw std::invalid_argument("cannot convert type to LLVM type");
        }
        llvm::Type* operator()(ast::FieldType type) const {
            auto elementType = operator()(type.elementType);
            auto ptrType = llvm::PointerType::get(elementType, 0);
            auto indexType = llvm::IntegerType::get(context, 8 * sizeof(size_t));
            auto indexArrayType = llvm::ArrayType::get(indexType, type.numDimensions);
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
        llvm::LLVMContext& context;
    };
    return std::visit(Visitor{ context }, type);
}

//------------------------------------------------------------------------------
// Argument
//------------------------------------------------------------------------------
Argument::Argument(ast::Type type, const Runner* runner)
    : m_type(type),
      m_runner(runner),
      m_llvmType(ConvertType(type, runner->GetContext())) {
}

//--------------------------------------
// Generic methods
//--------------------------------------

size_t Argument::GetSize() const {
    return std::visit([this](const auto& type) { return GetSize(type); }, m_type);
}

size_t Argument::GetAlignment() const {
    if (auto layout = GetLayout()) {
        return layout->getAlignment().value();
    }
    return GetSize();
}

pybind11::object Argument::Read(const void* address) {
    return std::visit([&, this](const auto& type) { return Read(type, address); }, m_type);
}

void Argument::Write(pybind11::object value, void* address) {
    return std::visit([&, this](const auto& type) { return Write(type, value, address); }, m_type);
}

//--------------------------------------
// ScalarType methods
//--------------------------------------
size_t Argument::GetSize(ast::ScalarType type) {
    return ast::VisitType(type, [](auto* typed) { return sizeof *typed; });
}

pybind11::object Argument::Read(ast::ScalarType type, const void* address) {
    return ast::VisitType(type, [&](auto* typed) -> pybind11::object {
        using T = std::decay_t<decltype(*typed)>;
        const T value = *static_cast<const T*>(address);
        if constexpr (std::is_integral_v<T>) {
            return pybind11::int_(value);
        }
        else if constexpr (std::is_floating_point_v<T>) {
            return pybind11::float_(value);
        }
        else {
            static_assert(!sizeof(T*), "scalar type not supported, this is an implementation error");
        }
    });
}

void Argument::Write(ast::ScalarType type, pybind11::object value, void* address) {
    ast::VisitType(type, [&](auto* typed) {
        using T = std::decay_t<decltype(*typed)>;
        *static_cast<T*>(address) = value.cast<T>();
    });
}

//--------------------------------------
// Field type methods
//--------------------------------------
size_t Argument::GetSize(ast::FieldType type) const {
    auto layout = GetLayout();
    assert(layout);
    return layout->getSizeInBytes();
}

pybind11::object Argument::Read(ast::FieldType type, const void* address) const {
    auto layout = GetLayout();
    assert(layout);
    return ast::VisitType(type.elementType, [&](auto* typed) -> pybind11::memoryview {
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
                       [](auto s) { return s * sizeof(T); });
        return pybind11::memoryview::from_buffer(*alignedPtrAddress + *offsetAddress,
                                                 std::span{ shapeAddress, type.numDimensions },
                                                 std::span{ byteStrides },
                                                 false);
    });
}

void Argument::Write(ast::FieldType type, pybind11::object value, void* address) const {
    auto layout = GetLayout();
    assert(layout);
    ast::VisitType(type.elementType, [&](auto* typed) {
        using T = std::decay_t<decltype(*typed)>;

        const auto buffer = value.cast<pybind11::buffer>();
        const auto bufferInfo = buffer.request(true);
        const auto bufferType = ast::FieldType{ GetTypeFromFormat(bufferInfo.format), size_t(bufferInfo.ndim) };
        if (bufferType != type) {
            std::stringstream ss;
            ss << "expected buffer type " << type << ", got " << bufferType;
            throw std::invalid_argument(ss.str());
        }

        const auto startingAddress = reinterpret_cast<std::byte*>(address);
        const auto allocPtrAddress = reinterpret_cast<T**>(startingAddress + layout->getElementOffset(0));
        const auto alignedPtrAddress = reinterpret_cast<T**>(startingAddress + layout->getElementOffset(1));
        const auto offsetAddress = reinterpret_cast<ptrdiff_t*>(startingAddress + layout->getElementOffset(2));
        const auto shapeAddress = reinterpret_cast<ptrdiff_t*>(startingAddress + layout->getElementOffset(3));
        const auto stridesAddress = reinterpret_cast<ptrdiff_t*>(startingAddress + layout->getElementOffset(4));

        *allocPtrAddress = reinterpret_cast<T*>(bufferInfo.ptr);
        *alignedPtrAddress = *allocPtrAddress;
        *offsetAddress = 0;
        std::copy(bufferInfo.shape.begin(), bufferInfo.shape.end(), shapeAddress);
        std::transform(bufferInfo.strides.begin(), bufferInfo.strides.end(), stridesAddress, [](auto s) { return s / sizeof(T); });
    });
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
ArgumentPack::ArgumentPack(std::span<const ast::Type> types, const Runner* runner)
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
