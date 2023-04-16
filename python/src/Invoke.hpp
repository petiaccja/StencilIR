#pragma once

#include <pybind11/pybind11.h>

#include <IR/Types.hpp>
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
    pybind11::object Read(const std::byte* address) const;
    void Write(pybind11::object value, std::byte* address) const;
    template <class Iter>
    void GetOpaquePointers(std::byte* address, Iter out) const;

    llvm::Type* GetLLVMType() const { return m_llvmType; }

private:
    pybind11::object Read(const ast::IntegerType& type, const std::byte* address) const;
    void Write(const ast::IntegerType& type, pybind11::object value, std::byte* address) const;
    template <class Iter>
    void GetOpaquePointers(const ast::IntegerType& type, std::byte* address, Iter out) const;

    pybind11::object Read(const ast::FloatType& type, const std::byte* address) const;
    void Write(const ast::FloatType& type, pybind11::object value, std::byte* address) const;
    template <class Iter>
    void GetOpaquePointers(const ast::FloatType& type, std::byte* address, Iter out) const;

    pybind11::object Read(const ast::IndexType& type, const std::byte* address) const;
    void Write(const ast::IndexType& type, pybind11::object value, std::byte* address) const;
    template <class Iter>
    void GetOpaquePointers(const ast::IndexType& type, std::byte* address, Iter out) const;

    pybind11::object Read(const ast::FieldType& type, const std::byte* address) const;
    void Write(const ast::FieldType& type, pybind11::object value, std::byte* address) const;
    template <class Iter>
    void GetOpaquePointers(const ast::FieldType& type, std::byte* address, Iter out) const;

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
    pybind11::object Read(const std::byte* address) const;
    void Write(pybind11::object value, std::byte* address) const;

    template <class Iter>
    void GetOpaquePointers(std::byte* address, Iter out) const;

private:
    const llvm::StructLayout* GetLayout() const;

private:
    std::vector<Argument> m_items;
    const Runner* m_runner;
    llvm::StructType* m_llvmType = nullptr;
};


template <class Iter>
void Argument::GetOpaquePointers(std::byte* address, Iter out) const {
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
void Argument::GetOpaquePointers(const ast::IntegerType&, std::byte* address, Iter out) const {
    *(out++) = address;
}

template <class Iter>
void Argument::GetOpaquePointers(const ast::FloatType&, std::byte* address, Iter out) const {
    *(out++) = address;
}

template <class Iter>
void Argument::GetOpaquePointers(const ast::IndexType&, std::byte* address, Iter out) const {
    *(out++) = address;
}

template <class Iter>
void Argument::GetOpaquePointers(const ast::FieldType& type, std::byte* address, Iter out) const {
    const auto startingAddress = address;
    const auto layout = GetLayout();
    *(out++) = startingAddress + layout->getElementOffset(0); // ptr
    *(out++) = startingAddress + layout->getElementOffset(1); // aligned ptr
    *(out++) = startingAddress + layout->getElementOffset(2); // offset
    for (size_t i = 0; i < type.numDimensions; ++i) {
        *(out++) = startingAddress + layout->getElementOffset(3) + i * sizeof(ptrdiff_t); // shape
    }
    for (size_t i = 0; i < type.numDimensions; ++i) {
        *(out++) = startingAddress + layout->getElementOffset(4) + i * sizeof(ptrdiff_t); // strides
    }
}

template <class Iter>
void ArgumentPack::GetOpaquePointers(std::byte* address, Iter out) const {
    const auto startingAddress = address;
    for (size_t i = 0; i < m_items.size(); ++i) {
        const auto& offset = GetLayout()->getElementOffset(i);
        m_items[i].GetOpaquePointers(startingAddress + offset, out);
    }
}