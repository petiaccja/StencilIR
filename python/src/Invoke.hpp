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


ast::TypePtr GetTypeFromFormat(std::string_view format);


class Argument {
public:
    Argument(ast::TypePtr type, const Runner* runner);

    size_t GetSize() const;
    size_t GetAlignment() const;
    pybind11::object Read(const void* address);
    void Write(pybind11::object value, void* address);
    template <class Iter>
    void GetOpaquePointers(void* address, Iter out) const;

    llvm::Type* GetLLVMType() const { return m_llvmType; }

private:
    pybind11::object Read(const ast::IntegerType& type, const void* address) const;
    void Write(const ast::IntegerType& type, pybind11::object value, void* address) const;
    template <class Iter>
    void GetOpaquePointers(const ast::IntegerType& type, void* address, Iter out) const;

    pybind11::object Read(const ast::FloatType& type, const void* address) const;
    void Write(const ast::FloatType& type, pybind11::object value, void* address) const;
    template <class Iter>
    void GetOpaquePointers(const ast::FloatType& type, void* address, Iter out) const;

    pybind11::object Read(const ast::IndexType& type, const void* address) const;
    void Write(const ast::IndexType& type, pybind11::object value, void* address) const;
    template <class Iter>
    void GetOpaquePointers(const ast::IndexType& type, void* address, Iter out) const;

    pybind11::object Read(const ast::FieldType& type, const void* address) const;
    void Write(const ast::FieldType& type, pybind11::object value, void* address) const;
    template <class Iter>
    void GetOpaquePointers(const ast::FieldType& type, void* address, Iter out) const;

    const llvm::StructLayout* GetLayout() const;

private:
    ast::TypePtr m_type;
    const Runner* m_runner = nullptr;
    llvm::Type* m_llvmType = nullptr;
};


class ArgumentPack {
public:
    ArgumentPack(std::span<const ast::TypePtr> types, const Runner* runner);

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
    if (auto type = dynamic_cast<const ast::IntegerType*>(m_type.get())) {
        return GetOpaquePointers(*type, address, out);
    }
    else if (auto type = dynamic_cast<const ast::FloatType*>(m_type.get())) {
        return GetOpaquePointers(*type, address, out);
    }
    else if (auto type = dynamic_cast<const ast::IndexType*>(m_type.get())) {
        return GetOpaquePointers(*type, address, out);
    }
    else if (auto type = dynamic_cast<const ast::FieldType*>(m_type.get())) {
        return GetOpaquePointers(*type, address, out);
    }
    std::terminate();
}

template <class Iter>
void Argument::GetOpaquePointers(const ast::IntegerType&, void* address, Iter out) const {
    *(out++) = address;
}

template <class Iter>
void Argument::GetOpaquePointers(const ast::FloatType&, void* address, Iter out) const {
    *(out++) = address;
}

template <class Iter>
void Argument::GetOpaquePointers(const ast::IndexType&, void* address, Iter out) const {
    *(out++) = address;
}

template <class Iter>
void Argument::GetOpaquePointers(const ast::FieldType& type, void* address, Iter out) const {
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