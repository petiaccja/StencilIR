#pragma once

#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/MemRefUtils.h>
#include <mlir/IR/BuiltinOps.h>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <tuple>


class Runner {
public:
    Runner(mlir::ModuleOp& llvmIr, int optLevel = 0);

    template <class... Args>
    void Invoke(std::string_view name, Args&&... args) const
        requires((... && !std::ranges::range<Args>));

    void Invoke(std::string_view name, std::span<void*> args) const;

    llvm::LLVMContext& GetContext() const;
    const llvm::DataLayout& GetDataLayout() const;
    std::string GetLLVMIR() const;
    std::vector<char> GetObjectFile() const;

private:
    std::unique_ptr<mlir::ExecutionEngine> m_engine;
    std::unique_ptr<llvm::LLVMContext> m_llvmContext;
    std::unique_ptr<llvm::Module> m_llvmModule;
};


namespace impl {
auto FlattenArg(const auto& v) {
    return std::tuple{ v };
}

template <class T, int... Dims>
auto FlattenArg(const StridedMemRefType<T, sizeof...(Dims)>& v, std::integer_sequence<int, Dims...>) {
    return std::tuple{
        v.basePtr,
        v.data,
        v.offset,
        v.sizes[Dims]...,
        v.strides[Dims]...,
    };
}

template <class T, int N>
auto FlattenArg(const StridedMemRefType<T, N>& v) {
    return FlattenArg(v, std::make_integer_sequence<int, N>{});
}
} // namespace impl


template <class... Args>
void Runner::Invoke(std::string_view name, Args&&... args) const
    requires((... && !std::ranges::range<Args>))
{
    auto flattenedArgs = std::tuple_cat(impl::FlattenArg(std::forward<Args>(args))...);
    auto opaqueArgs = std::apply([](auto&&... args) { return std::array{ static_cast<void*>(&args)... }; }, flattenedArgs);
    Invoke(name, opaqueArgs);
}