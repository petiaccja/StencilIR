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


namespace sir {


class Runner {
public:
    template <class T>
    struct Result {
        Result(T& value) : ptr(&value) {}
        T* ptr;
    };

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

    template <class T>
    auto FlattenArg(const Runner::Result<T>& result) {
        return std::tuple{};
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

    auto FlattenResult(const auto& v) {
        return std::tuple{};
    }

    template <class T>
    auto FlattenResult(const Runner::Result<T>& v) {
        return std::tuple{ v.ptr };
    }
} // namespace impl


template <class... Args>
void Runner::Invoke(std::string_view name, Args&&... args) const
    requires((... && !std::ranges::range<Args>))
{
    auto flattenedArgs = std::tuple_cat(impl::FlattenArg(std::forward<Args>(args))...);
    auto flattenedResults = std::tuple_cat(impl::FlattenResult(std::forward<Args>(args))...);
    auto opaqueArgs = std::apply([](auto&&... args) {
        using ArrayT = std::array<void*, std::tuple_size_v<decltype(flattenedArgs)>>;
        return ArrayT{ static_cast<void*>(&args)... };
    },
                                 flattenedArgs);
    auto opaqueResults = std::apply([](auto&&... results) {
        using ArrayT = std::array<void*, std::tuple_size_v<decltype(flattenedResults)>>;
        return ArrayT{ static_cast<void*>(results)... };
    },
                                    flattenedResults);
    std::array<void*, std::ssize(opaqueArgs) + std::ssize(opaqueResults)> opaqueAll;
    const auto it = std::copy(opaqueArgs.begin(), opaqueArgs.end(), opaqueAll.begin());
    std::copy(opaqueResults.begin(), opaqueResults.end(), it);
    Invoke(name, opaqueAll);
}


} // namespace sir