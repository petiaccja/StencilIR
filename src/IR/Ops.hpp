#pragma once

#include "Operation.hpp"


namespace sir::ops {


//------------------------------------------------------------------------------
// Enumerations
//------------------------------------------------------------------------------

enum class eArithmeticFunction {
    ADD,
    SUB,
    MUL,
    DIV,
    MOD,
    BIT_AND,
    BIT_OR,
    BIT_XOR,
    BIT_SHL,
    BIT_SHR,
};


enum class eComparisonFunction {
    EQ,
    NEQ,
    LT,
    GT,
    LTE,
    GTE,
};


//------------------------------------------------------------------------------
// Attributes
//------------------------------------------------------------------------------

struct FuncAttr {
    std::string name;
    std::shared_ptr<FunctionType> signature;
    bool isPublic;
};


struct StencilAttr {
    std::string name;
    std::shared_ptr<FunctionType> signature;
    int numDims;
    bool isPublic;
};


struct CallAttr {
    std::string name;
    std::vector<TypePtr> results;
};


struct ApplyAttr {
    std::string name;
    size_t numInputs;
    size_t numOutputs;
    size_t numOffsets;
    std::vector<int64_t> staticOffsets;
};


struct ConstantAttr {
    TypePtr type;
    std::any value;
};


//------------------------------------------------------------------------------
// Common operation types & utilities
//------------------------------------------------------------------------------

struct SingleRegion : Operation {
    SingleRegion(std::type_index type,
                 std::vector<Value> operands,
                 size_t numResults,
                 std::any attributes,
                 std::optional<sir::Location> loc = {})
        : Operation(type, operands, numResults, { Region{} }, attributes, loc) {}

    Region& GetBody() { return GetRegions().front(); }
    const Region& GetBody() const { return GetRegions().front(); }

    size_t GetNumRegionArgs() const { return GetBody().GetArgs().size(); }
    const auto& GetRegionArgs() const { return GetBody().GetArgs(); }
    const Value GetRegionArg(size_t index) const {
        assert(GetNumRegionArgs() > index);
        return GetRegionArgs()[index];
    }

    template <class OpT, class... Args>
    OpT Create(Args&&... args) {
        auto op = OpT(std::forward<Args>(args)...);
        GetBody().GetOperations().push_back(op);
        return op;
    }
};


template <class Vector, class... Vectors>
auto ConcatVectors(const Vector& head, const Vectors&... rest) {
    auto result = head;
    (..., std::ranges::copy(rest, std::back_inserter(result)));
    return result;
}


//------------------------------------------------------------------------------
// Module & function organization
//------------------------------------------------------------------------------

struct ModuleOp : SingleRegion {
    ModuleOp(std::optional<sir::Location> loc = {})
        : SingleRegion(typeid(decltype(*this)), {}, 0, {}, loc) {}
};


struct FuncOp : SingleRegion {
    FuncOp(std::string name,
           std::shared_ptr<FunctionType> signature,
           bool isPublic = true,
           std::optional<sir::Location> loc = {})
        : SingleRegion(typeid(decltype(*this)), {}, 0, FuncAttr{ name, signature, isPublic }, loc) {
        size_t index = 0;
        for (auto type : signature->parameters) {
            GetBody().GetArgs().push_back(Value(*this, index++));
        }
    }

    std::string_view GetName() const {
        return std::any_cast<const FuncAttr&>(GetAttributes()).name;
    }

    std::shared_ptr<FunctionType> GetFunctionType() const {
        return std::any_cast<const FuncAttr&>(GetAttributes()).signature;
    }
};


struct StencilOp : SingleRegion {
    StencilOp(std::string name,
              std::shared_ptr<FunctionType> signature,
              int numDims,
              bool isPublic = true,
              std::optional<sir::Location> loc = {})
        : SingleRegion(typeid(decltype(*this)), {}, 0, StencilAttr{ name, signature, numDims, isPublic }, loc) {
        size_t index = 0;
        for (auto type : signature->parameters) {
            GetBody().GetArgs().push_back(Value(*this, index++));
        }
    }

    std::string_view GetName() const {
        return std::any_cast<const StencilAttr&>(GetAttributes()).name;
    }

    std::shared_ptr<FunctionType> GetFunctionType() const {
        return std::any_cast<const StencilAttr&>(GetAttributes()).signature;
    }
};


struct ReturnOp : Operation {
    ReturnOp(std::vector<Value> values, std::optional<sir::Location> loc = {})
        : Operation(typeid(decltype(*this)), values, 0, {}, {}, loc) {}

    auto GetValues() const { return GetOperands(); }
};


struct CallOp : Operation {
    CallOp(FuncOp callee, std::vector<Value> args, std::optional<sir::Location> loc = {})
        : CallOp(std::string{ callee.GetName() }, callee.GetFunctionType()->results, std::move(args), std::move(loc)) {}

    CallOp(std::string callee, std::vector<TypePtr> results, std::vector<Value> args, std::optional<sir::Location> loc = {})
        : Operation(typeid(decltype(*this)),
                    args,
                    results.size(),
                    {},
                    CallAttr{ callee, results },
                    loc) {}

    std::string GetCallee() const { return std::any_cast<const CallAttr&>(GetAttributes()).name; }
    size_t GetNumResults() const { return GetResults().size(); }
    auto GetArgs() const { return GetOperands(); }
};


struct ApplyOp : Operation {
    ApplyOp(StencilOp stencil,
            std::vector<Value> inputs,
            std::vector<Value> outputs,
            std::vector<Value> offsets,
            std::vector<int64_t> staticOffsets,
            std::optional<sir::Location> loc = {})
        : ApplyOp(std::string(stencil.GetName()),
                  std::move(inputs),
                  std::move(outputs),
                  std::move(offsets),
                  std::move(staticOffsets),
                  std::move(loc)) {}

    ApplyOp(std::string stencil,
            std::vector<Value> inputs,
            std::vector<Value> outputs,
            std::vector<Value> offsets,
            std::vector<int64_t> staticOffsets,
            std::optional<sir::Location> loc = {})
        : Operation(typeid(decltype(*this)),
                    ConcatVectors(inputs, outputs, offsets),
                    outputs.size(),
                    {},
                    ApplyAttr{ stencil, inputs.size(), outputs.size(), offsets.size(), staticOffsets },
                    loc) {}

    std::string GetStencil() const { return std::any_cast<const ApplyAttr&>(GetAttributes()).name; }
    size_t GetNumResults() const { return GetResults().size(); }
    size_t GetNumInputs() const { return std::any_cast<const ApplyAttr&>(GetAttributes()).numInputs; }
    size_t GetNumOutputs() const { return std::any_cast<const ApplyAttr&>(GetAttributes()).numOutputs; }
    size_t GetNumOffsets() const { return std::any_cast<const ApplyAttr&>(GetAttributes()).numOffsets; }
    auto GetInputs() const { return GetOperands().subspan(0, GetNumInputs()); }
    auto GetOutputs() const { return GetOperands().subspan(GetNumInputs(), GetNumOutputs()); }
    auto GetOffsets() const { return GetOperands().subspan(GetNumInputs() + GetNumOutputs(), GetNumOffsets()); }
    auto GetStaticOffsets() const { return std::any_cast<const ApplyAttr&>(GetAttributes()).staticOffsets; }
};


struct InvokeOp;


//------------------------------------------------------------------------------
// Arithmetic-logic
//------------------------------------------------------------------------------

struct CastOp : Operation {
    CastOp(Value input, TypePtr type, std::optional<sir::Location> loc = {})
        : Operation(typeid(decltype(*this)), { input }, 1, {}, type, loc) {}
    Value GetInput() const { return GetOperands()[0].GetSource(); }
    TypePtr GetType() const { return std::any_cast<TypePtr>(GetAttributes()); }
    Value GetResult() const { return GetResults()[0]; }
};


struct ConstantOp : Operation {
    template <class T>
    explicit ConstantOp(T value, std::optional<sir::Location> loc = {})
        : ConstantOp(std::move(value), InferType<std::decay_t<T>>(), std::move(loc)) {}
    explicit ConstantOp(auto value, TypePtr type, std::optional<sir::Location> loc = {})
        : Operation(typeid(decltype(*this)), {}, 1, {}, ConstantAttr{ type, WrapValue(value, type) }, loc) {}

    auto GetValue() const { return std::any_cast<const ConstantAttr&>(GetAttributes()).value; }
    TypePtr GetType() const { return std::any_cast<const ConstantAttr&>(GetAttributes()).type; }
    Value GetResult() const { return GetResults()[0]; }

private:
    static std::any WrapValue(auto value, TypePtr type) {
        if (std::dynamic_pointer_cast<IntegerType>(type)) {
            return static_cast<int64_t>(value);
        }
        else if (std::dynamic_pointer_cast<IndexType>(type)) {
            return static_cast<int64_t>(value);
        }
        else if (std::dynamic_pointer_cast<FloatType>(type)) {
            return static_cast<double>(value);
        }
        else {
            return value;
        }
    }
};

struct ArithmeticOp : Operation {
    ArithmeticOp(Value lhs, Value rhs, eArithmeticFunction function, std::optional<sir::Location> loc = {})
        : Operation(typeid(decltype(*this)), { lhs, rhs }, 1, {}, function, loc) {}
    Value GetLeft() const { return GetOperands()[0].GetSource(); }
    Value GetRight() const { return GetOperands()[1].GetSource(); }
    eArithmeticFunction GetFunction() const { return std::any_cast<eArithmeticFunction>(GetAttributes()); }
    Value GetResult() const { return GetResults()[0]; }
};


struct ComparisonOp : Operation {
    ComparisonOp(Value lhs, Value rhs, eComparisonFunction function, std::optional<sir::Location> loc = {})
        : Operation(typeid(decltype(*this)), { lhs, rhs }, 1, {}, function, loc) {}
    Value GetLeft() const { return GetOperands()[0].GetSource(); }
    Value GetRight() const { return GetOperands()[1].GetSource(); }
    eComparisonFunction GetFunction() const { return std::any_cast<eComparisonFunction>(GetAttributes()); }
    Value GetResult() const { return GetResults()[0]; }
};


struct MinOp : Operation {
    MinOp(Value lhs, Value rhs, std::optional<sir::Location> loc = {})
        : Operation(typeid(decltype(*this)), { lhs, rhs }, 1, {}, {}, loc) {}
    Value GetLeft() const { return GetOperands()[0].GetSource(); }
    Value GetRight() const { return GetOperands()[1].GetSource(); }
    Value GetResult() const { return GetResults()[0]; }
};


struct MaxOp : Operation {
    MaxOp(Value lhs, Value rhs, std::optional<sir::Location> loc = {})
        : Operation(typeid(decltype(*this)), { lhs, rhs }, 1, {}, {}, loc) {}
    Value GetLeft() const { return GetOperands()[0].GetSource(); }
    Value GetRight() const { return GetOperands()[1].GetSource(); }
    Value GetResult() const { return GetResults()[0]; }
};


//------------------------------------------------------------------------------
// Control flow
//------------------------------------------------------------------------------

struct IfOp : Operation {
    IfOp(Value cond, size_t numResults, std::optional<sir::Location> loc = {})
        : Operation(typeid(decltype(*this)), { cond }, numResults, { Region(), Region() }, {}, loc) {}

    Value GetCondition() const { return GetOperands()[0].GetSource(); }

    Region& GetThenRegion() { return GetRegions()[0]; }
    const Region& GetThenRegion() const { return GetRegions()[0]; }

    Region& GetElseRegion() { return GetRegions()[1]; }
    const Region& GetElseRegion() const { return GetRegions()[1]; }
};


struct ForOp : SingleRegion {
    ForOp(Value start, Value stop, Value step, std::vector<Value> init, std::optional<sir::Location> loc = {})
        : SingleRegion(typeid(decltype(*this)),
                       ConcatVectors(std::vector{ start, stop, step }, init),
                       init.size(),
                       {},
                       loc) {
        GetBody().GetArgs().push_back(Value(*this, 0)); // Loop index
        for (size_t index = 0; index < init.size(); ++index) { // Loop carried vars
            GetBody().GetArgs().push_back(Value(*this, 1 + index));
        }
    }

    Value GetStart() const { return GetOperands()[0].GetSource(); }
    Value GetStop() const { return GetOperands()[1].GetSource(); }
    Value GetStep() const { return GetOperands()[2].GetSource(); }
};


struct YieldOp : Operation {
    YieldOp(std::vector<Value> values, std::optional<sir::Location> loc = {})
        : Operation(typeid(decltype(*this)), values, 0, {}, {}, loc) {}

    auto GetValues() { return GetOperands(); }
};


//------------------------------------------------------------------------------
// Tensor
//------------------------------------------------------------------------------

struct DimOp : Operation {
    DimOp(Value source, Value index, std::optional<sir::Location> loc = {})
        : Operation(typeid(decltype(*this)), { source, index }, 1, {}, {}, loc) {}

    auto GetResult() const { return GetResults()[0]; }
};


struct AllocTensorOp : Operation {
    AllocTensorOp(TypePtr elementType, std::vector<Value> sizes, std::optional<sir::Location> loc = {})
        : Operation(typeid(decltype(*this)), sizes, 1, {}, elementType, loc) {}

    auto GetResult() const { return GetResults()[0]; }
};


struct ExtractSliceOp : Operation {
    ExtractSliceOp(Value source,
                   std::vector<Value> offsets,
                   std::vector<Value> sizes,
                   std::vector<Value> strides,
                   std::optional<sir::Location> loc = {})
        : Operation(typeid(decltype(*this)),
                    ConcatVectors(std::vector<Value>{ source }, offsets, sizes, strides),
                    1,
                    {},
                    {},
                    loc) {}

    auto GetResult() const { return GetResults()[0]; }
};


struct InsertSliceOp : Operation {
    InsertSliceOp(Value source,
                  Value dest,
                  std::vector<Value> offsets,
                  std::vector<Value> sizes,
                  std::vector<Value> strides,
                  std::optional<sir::Location> loc = {})
        : Operation(typeid(decltype(*this)),
                    ConcatVectors(std::vector<Value>{ source, dest }, offsets, sizes, strides),
                    1,
                    {},
                    {},
                    loc) {}

    auto GetResult() const { return GetResults()[0]; }
};


//------------------------------------------------------------------------------
// Stencil
//------------------------------------------------------------------------------

struct IndexOp : Operation {
    IndexOp(std::optional<sir::Location> loc = {})
        : Operation(typeid(decltype(*this)), {}, 1, {}, {}, loc) {}

    auto GetResult() const { return GetResults()[0]; }
};


struct JumpOp : Operation {
    JumpOp(Value index, std::vector<int64_t> offsets, std::optional<sir::Location> loc = {})
        : Operation(typeid(decltype(*this)), { index }, 1, {}, offsets, loc) {}

    auto GetResult() const { return GetResults()[0]; }
};


struct ProjectOp : Operation {
    ProjectOp(Value index, std::vector<int64_t> positions, std::optional<sir::Location> loc = {})
        : Operation(typeid(decltype(*this)), { index }, 1, {}, positions, loc) {}

    auto GetResult() const { return GetResults()[0]; }
};


struct ExtendOp : Operation {
    ExtendOp(Value index, int64_t position, Value value, std::optional<sir::Location> loc = {})
        : Operation(typeid(decltype(*this)), { index, value }, 1, {}, position, loc) {}

    auto GetResult() const { return GetResults()[0]; }
};


struct ExchangeOp : Operation {
    ExchangeOp(Value index, int64_t position, Value value, std::optional<sir::Location> loc = {})
        : Operation(typeid(decltype(*this)), { index, value }, 1, {}, position, loc) {}

    auto GetResult() const { return GetResults()[0]; }
};


struct ExtractOp : Operation {
    ExtractOp(Value index, int64_t position, std::optional<sir::Location> loc = {})
        : Operation(typeid(decltype(*this)), { index }, 1, {}, position, loc) {}

    auto GetResult() const { return GetResults()[0]; }
};


struct SampleOp : Operation {
    SampleOp(Value tensor, Value index, std::optional<sir::Location> loc = {})
        : Operation(typeid(decltype(*this)), { tensor, index }, 1, {}, {}, loc) {}

    auto GetResult() const { return GetResults()[0]; }
};


} // namespace sir::ops