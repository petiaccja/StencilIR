#pragma once

#include <pybind11/pybind11.h>

#include <AST/Types.hpp>
#include <Diagnostics/Exception.hpp>
#include <Execution/Execution.hpp>

#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Type.h>

#include <concepts>
#include <memory>
#include <memory_resource>
#include <vector>


ast::ScalarType GetTypeFromFormat(std::string_view format);


class Argument {
public:
    Argument(ast::Type type, const Runner* runner);

    size_t GetSize() const;
    size_t GetAlignment() const;
    pybind11::object Read(const void* address);
    void Write(pybind11::object value, void* address);
    template <class Iter>
    void GetOpaquePointers(void* address, Iter out) const;

    llvm::Type* GetLLVMType() const { return m_llvmType; }

private:
    static size_t GetSize(ast::ScalarType type);
    static pybind11::object Read(ast::ScalarType type, const void* address);
    static void Write(ast::ScalarType type, pybind11::object value, void* address);
    template <class Iter>
    static void GetOpaquePointers(ast::ScalarType, void* address, Iter out);

    size_t GetSize(ast::FieldType type) const;
    pybind11::object Read(ast::FieldType type, const void* address) const;
    void Write(ast::FieldType type, pybind11::object value, void* address) const;
    template <class Iter>
    void GetOpaquePointers(ast::FieldType, void* address, Iter out) const;

    const llvm::StructLayout* GetLayout() const;

private:
    ast::Type m_type;
    const Runner* m_runner = nullptr;
    llvm::Type* m_llvmType = nullptr;
};


class ArgumentPack {
public:
    ArgumentPack(std::span<const ast::Type> types, const Runner* runner);

    size_t GetSize() const;
    size_t GetAlignment() const;
    pybind11::object Read(const void* address);
    void Write(pybind11::object value, void* address);

    template <class Iter>
    void GetOpaquePointers(void* address, Iter out) const;

private:
    const llvm::StructLayout* GetLayout() const;

private:
    std::vector<Argument> m_items;
    const Runner* m_runner;
    llvm::StructType* m_llvmType = nullptr;
};


template <class Iter>
void Argument::GetOpaquePointers(void* address, Iter out) const {
    std::visit([&, this](auto type) { GetOpaquePointers(type, address, out); }, m_type);
}

template <class Iter>
void Argument::GetOpaquePointers(ast::ScalarType, void* address, Iter out) {
    *(out++) = address;
}

template <class Iter>
void Argument::GetOpaquePointers(ast::FieldType type, void* address, Iter out) const {
    const auto startingAddress = reinterpret_cast<std::byte*>(address);
    const auto layout = GetLayout();
    *(out++) = reinterpret_cast<void*>(startingAddress + layout->getElementOffset(0)); // ptr
    *(out++) = reinterpret_cast<void*>(startingAddress + layout->getElementOffset(1)); // aligned ptr
    *(out++) = reinterpret_cast<void*>(startingAddress + layout->getElementOffset(2)); // offset
    for (size_t i = 0; i < type.numDimensions; ++i) {
        *(out++) = reinterpret_cast<void*>(startingAddress + layout->getElementOffset(3) + i * sizeof(ptrdiff_t)); // shape
    }
    for (size_t i = 0; i < type.numDimensions; ++i) {
        *(out++) = reinterpret_cast<void*>(startingAddress + layout->getElementOffset(4) + i * sizeof(ptrdiff_t)); // strides
    }
}

template <class Iter>
void ArgumentPack::GetOpaquePointers(void* address, Iter out) const {
    const auto startingAddress = reinterpret_cast<std::byte*>(address);
    for (size_t i = 0; i < m_items.size(); ++i) {
        const auto& offset = GetLayout()->getElementOffset(i);
        m_items[i].GetOpaquePointers(reinterpret_cast<void*>(startingAddress + offset), out);
    }
}