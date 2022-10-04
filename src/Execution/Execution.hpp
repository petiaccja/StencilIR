#pragma once

#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/IR/BuiltinOps.h>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>
#include <tuple>


template <class T, size_t Dim>
struct MemRef {
    T* ptr;
    T* alignedPtr;
    ptrdiff_t offset;
    std::array<ptrdiff_t, Dim> shape;
    std::array<ptrdiff_t, Dim> strides;
};


class Runner {
public:
    Runner(mlir::ModuleOp& llvmIr, int optLevel = 0);

    template <class... Args>
    void Invoke(std::string_view name, Args&&... args) const
        requires((... && !std::ranges::range<Args>));
    void Invoke(std::string_view name, std::span<void*> args) const;

    template <class Arg>
    static auto MakeCompatibleArgument(const Arg& arg) {
        return ConvertArgs(arg);
    }

    template <class CompatibleArg>
    static auto MakeOpaqueArgument(CompatibleArg& arg) {
        return OpaqueArgs(arg);
    }

    std::string_view LLVMIR() const { return m_llvmIrDump; }

private:
    static auto ConvertArg(const std::floating_point auto& arg) {
        return std::tuple{ arg };
    }

    static auto ConvertArg(const std::integral auto& arg) {
        return std::tuple{ arg };
    }

    static auto ConvertArg(const auto* arg) {
        return std::tuple{ arg };
    }

    template <class T, size_t Dim, size_t... Indices>
    static auto ArrayToTupleHelper(const std::array<T, Dim>& arr, std::index_sequence<Indices...>) {
        return std::make_tuple(arr[Indices]...);
    }

    template <class T, size_t Dim>
    static auto ArrayToTuple(const std::array<T, Dim>& arr) {
        return ArrayToTupleHelper(arr, std::make_index_sequence<Dim>());
    }

    template <class T, size_t Dim>
    static auto ConvertArg(const MemRef<T, Dim>& arg) {
        return std::tuple_cat(std::tuple{ arg.ptr, arg.alignedPtr, arg.offset },
                              ArrayToTuple(arg.shape),
                              ArrayToTuple(arg.strides));
    }

    template <class... Args>
    static auto ConvertArgs(const Args&... args) {
        return std::tuple_cat(ConvertArg(args)...);
    }

    template <class... Args, size_t... Indices>
    static auto OpaqueArgsHelper(std::tuple<Args...>& args, std::index_sequence<Indices...>) {
        return std::array{ static_cast<void*>(std::addressof(std::get<Indices>(args)))... };
    }

    template <class... Args>
    static auto OpaqueArgs(std::tuple<Args...>& args) {
        return OpaqueArgsHelper(args, std::make_index_sequence<sizeof...(Args)>());
    }

private:
    std::unique_ptr<mlir::ExecutionEngine> m_engine;
    std::string m_llvmIrDump;
};


template <class... Args>
void Runner::Invoke(std::string_view name, Args&&... args) const
    requires((... && !std::ranges::range<Args>)) {
    auto convertedArgs = ConvertArgs(args...);
    std::array opaqueArgs = OpaqueArgs(convertedArgs);
    Invoke(name, opaqueArgs);
}