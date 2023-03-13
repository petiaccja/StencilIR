#include "Operation.hpp"


namespace dag {

// Needs tests:
//
// IndexOp
// JumpOp
// ExtendOp
// ExchangeOp
// ExtractOp
// SampleOp

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
    std::shared_ptr<ast::FunctionType> signature;
    bool isPublic;
};


struct StencilAttr {
    std::string name;
    std::shared_ptr<ast::FunctionType> signature;
    int numDims;
    bool isPublic;
};

struct CallAttr {
    std::string name;
};

struct ApplyAttr {
    std::string name;
    size_t numInputs;
    size_t numOutputs;
    size_t numOffsets;
    std::vector<int64_t> staticOffsets;
};


struct ConstantAttr {
    ast::TypePtr type;
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
                 std::optional<dag::Location> loc = {})
        : Operation(type, operands, numResults, { Region{} }, attributes, loc) {}

    Region& GetBody() { return Regions().front(); }
    const Region& GetBody() const { return Regions().front(); }

    size_t GetNumRegionArgs() const { return GetBody().args.size(); }
    const auto& GetRegionArgs() const { return GetBody().args; }
    const Value GetRegionArg(size_t index) const {
        assert(GetNumRegionArgs() > index);
        return GetRegionArgs()[index];
    }

    template <class OpT, class... Args>
    OpT Create(Args&&... args) {
        auto op = OpT(std::forward<Args>(args)...);
        GetBody().operations.push_back(op);
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
    ModuleOp(std::optional<dag::Location> loc = {})
        : SingleRegion(typeid(decltype(*this)), {}, 0, {}, loc) {}
};


struct FuncOp : SingleRegion {
    FuncOp(std::string name,
           std::shared_ptr<ast::FunctionType> signature,
           bool isPublic = true,
           std::optional<dag::Location> loc = {})
        : SingleRegion(typeid(decltype(*this)), {}, 0, FuncAttr{ name, signature, isPublic }, loc) {
        size_t index = 0;
        for (auto type : signature->parameters) {
            GetBody().args.push_back(Value(*this, index++));
        }
    }

    std::string_view GetName() const {
        return std::any_cast<const FuncAttr&>(Attributes()).name;
    }

    std::shared_ptr<ast::FunctionType> GetFunctionType() const {
        return std::any_cast<const FuncAttr&>(Attributes()).signature;
    }
};


struct StencilOp : SingleRegion {
    StencilOp(std::string name,
              std::shared_ptr<ast::FunctionType> signature,
              int numDims,
              bool isPublic = true,
              std::optional<dag::Location> loc = {})
        : SingleRegion(typeid(decltype(*this)), {}, 0, StencilAttr{ name, signature, numDims, isPublic }, loc) {
        size_t index = 0;
        for (auto type : signature->parameters) {
            GetBody().args.push_back(Value(*this, index++));
        }
    }

    std::string_view GetName() const {
        return std::any_cast<const StencilAttr&>(Attributes()).name;
    }

    std::shared_ptr<ast::FunctionType> GetFunctionType() const {
        return std::any_cast<const StencilAttr&>(Attributes()).signature;
    }
};


struct ReturnOp : Operation {
    ReturnOp(std::vector<Value> values, std::optional<dag::Location> loc = {})
        : Operation(typeid(decltype(*this)), values, 0, {}, {}, loc) {}
};


struct CallOp : Operation {
    CallOp(FuncOp func, std::vector<Value> args, std::optional<dag::Location> loc = {})
        : Operation(typeid(decltype(*this)),
                    args,
                    func.GetFunctionType()->results.size(),
                    {},
                    CallAttr{ std::string(func.GetName()) },
                    loc) {}
};


struct ApplyOp : Operation {
    ApplyOp(StencilOp stencil,
            std::vector<Value> inputs,
            std::vector<Value> outputs,
            std::vector<Value> offsets,
            std::vector<int64_t> staticOffsets,
            std::optional<dag::Location> loc = {})
        : Operation(typeid(decltype(*this)),
                    ConcatVectors(inputs, outputs, offsets),
                    stencil.GetFunctionType()->results.size(),
                    {},
                    ApplyAttr{ std::string{ stencil.GetName() }, inputs.size(), outputs.size(), offsets.size(), staticOffsets },
                    loc) {}
};


struct InvokeOp;


//------------------------------------------------------------------------------
// Arithmetic-logic
//------------------------------------------------------------------------------

struct CastOp : Operation {
    CastOp(Value input, ast::TypePtr type, std::optional<dag::Location> loc = {})
        : Operation(typeid(decltype(*this)), { input }, 1, {}, type, loc) {}
    Value Input() const { return Operands()[0].Source(); }
    Value Result() const { return Results()[0]; }
};


struct ConstantOp : Operation {
    template <class T>
    explicit ConstantOp(T value, std::optional<dag::Location> loc = {})
        : ConstantOp(std::move(value), ast::InferType<std::decay_t<T>>(), std::move(loc)) {}
    explicit ConstantOp(auto value, ast::TypePtr type, std::optional<dag::Location> loc = {})
        : Operation(typeid(decltype(*this)), {}, 1, {}, ConstantAttr{ type, WrapValue(value, type) }, loc) {}

    Value Result() const { return Results()[0]; }

private:
    static std::any WrapValue(auto value, ast::TypePtr type) {
        if (std::dynamic_pointer_cast<ast::IntegerType>(type)) {
            return static_cast<int64_t>(value);
        }
        else if (std::dynamic_pointer_cast<ast::IndexType>(type)) {
            return static_cast<int64_t>(value);
        }
        else if (std::dynamic_pointer_cast<ast::FloatType>(type)) {
            return static_cast<double>(value);
        }
        else {
            return value;
        }
    }
};

struct ArithmeticOp : Operation {
    ArithmeticOp(Value lhs, Value rhs, eArithmeticFunction function, std::optional<dag::Location> loc = {})
        : Operation(typeid(decltype(*this)), { lhs, rhs }, 1, {}, function, loc) {}
    Value Lhs() const { return Operands()[0].Source(); }
    Value Rhs() const { return Operands()[1].Source(); }
    Value Result() const { return Results()[0]; }
};


struct ComparisonOp : Operation {
    ComparisonOp(Value lhs, Value rhs, eComparisonFunction function, std::optional<dag::Location> loc = {})
        : Operation(typeid(decltype(*this)), { lhs, rhs }, 1, {}, function, loc) {}
    Value Lhs() const { return Operands()[0].Source(); }
    Value Rhs() const { return Operands()[1].Source(); }
    Value Result() const { return Results()[0]; }
};


struct MinOp : Operation {
    MinOp(Value lhs, Value rhs, std::optional<dag::Location> loc = {})
        : Operation(typeid(decltype(*this)), { lhs, rhs }, 1, {}, {}, loc) {}
    Value Lhs() const { return Operands()[0].Source(); }
    Value Rhs() const { return Operands()[1].Source(); }
    Value Result() const { return Results()[0]; }
};


struct MaxOp : Operation {
    MaxOp(Value lhs, Value rhs, std::optional<dag::Location> loc = {})
        : Operation(typeid(decltype(*this)), { lhs, rhs }, 1, {}, {}, loc) {}
    Value Lhs() const { return Operands()[0].Source(); }
    Value Rhs() const { return Operands()[1].Source(); }
    Value Result() const { return Results()[0]; }
};


//------------------------------------------------------------------------------
// Control flow
//------------------------------------------------------------------------------

struct IfOp : Operation {
    IfOp(Value cond, size_t numResults, std::optional<dag::Location> loc = {})
        : Operation(typeid(decltype(*this)), { cond }, numResults, { Region(), Region() }, {}, loc) {}

    Region& GetThenRegion() { return Regions()[0]; }
    const Region& GetThenRegion() const { return Regions()[0]; }

    Region& GetElseRegion() { return Regions()[1]; }
    const Region& GetElseRegion() const { return Regions()[1]; }
};


struct ForOp : SingleRegion {
    ForOp(Value start, Value stop, Value step, std::vector<Value> init, std::optional<dag::Location> loc = {})
        : SingleRegion(typeid(decltype(*this)),
                       ConcatVectors(std::vector{ start, stop, step }, init),
                       init.size(),
                       {},
                       loc) {
        GetBody().args.push_back(Value(*this, 0)); // Loop index
        for (size_t index = 0; index < init.size(); ++index) { // Loop carried vars
            GetBody().args.push_back(Value(*this, 1 + index++));
        }
    }
};


struct YieldOp : Operation {
    YieldOp(std::vector<Value> values, std::optional<dag::Location> loc = {})
        : Operation(typeid(decltype(*this)), values, 0, {}, {}, loc) {}
};


//------------------------------------------------------------------------------
// Tensor
//------------------------------------------------------------------------------

struct DimOp : Operation {
    DimOp(Value source, Value index, std::optional<dag::Location> loc = {})
        : Operation(typeid(decltype(*this)), { source, index }, 1, {}, {}, loc) {}
};


struct AllocTensorOp : Operation {
    AllocTensorOp(ast::TypePtr elementType, std::vector<Value> sizes, std::optional<dag::Location> loc = {})
        : Operation(typeid(decltype(*this)), sizes, 1, {}, elementType, loc) {}
};


struct ExtractSliceOp : Operation {
    ExtractSliceOp(Value source,
                   std::vector<Value> offsets,
                   std::vector<Value> sizes,
                   std::vector<Value> strides,
                   std::optional<dag::Location> loc = {})
        : Operation(typeid(decltype(*this)),
                    ConcatVectors(std::vector<Value>{ source }, offsets, sizes, strides),
                    1,
                    {},
                    {},
                    loc) {}
};


struct InsertSliceOp : Operation {
    InsertSliceOp(Value source,
                  Value dest,
                  std::vector<Value> offsets,
                  std::vector<Value> sizes,
                  std::vector<Value> strides,
                  std::optional<dag::Location> loc = {})
        : Operation(typeid(decltype(*this)),
                    ConcatVectors(std::vector<Value>{ source, dest }, offsets, sizes, strides),
                    1,
                    {},
                    {},
                    loc) {}
};


//------------------------------------------------------------------------------
// Stencil
//------------------------------------------------------------------------------

struct IndexOp : Operation {
    IndexOp(std::optional<dag::Location> loc = {})
        : Operation(typeid(decltype(*this)), {}, 1, {}, {}, loc) {}

    auto Result() const { return Results()[0]; }
};


struct JumpOp : Operation {
    JumpOp(Value index, std::vector<int64_t> offsets, std::optional<dag::Location> loc = {})
        : Operation(typeid(decltype(*this)), { index }, 1, {}, offsets, loc) {}

    auto Result() const { return Results()[0]; }
};


struct ProjectOp : Operation {
    ProjectOp(Value index, std::vector<int64_t> positions, std::optional<dag::Location> loc = {})
        : Operation(typeid(decltype(*this)), { index }, 1, {}, positions, loc) {}

    auto Result() const { return Results()[0]; }
};


struct ExtendOp : Operation {
    ExtendOp(Value index, int64_t position, Value value, std::optional<dag::Location> loc = {})
        : Operation(typeid(decltype(*this)), { index, value }, 1, {}, position, loc) {}

    auto Result() const { return Results()[0]; }
};


struct ExchangeOp : Operation {
    ExchangeOp(Value index, int64_t position, Value value, std::optional<dag::Location> loc = {})
        : Operation(typeid(decltype(*this)), { index, value }, 1, {}, position, loc) {}

    auto Result() const { return Results()[0]; }
};


struct ExtractOp : Operation {
    ExtractOp(Value index, int64_t position, std::optional<dag::Location> loc = {})
        : Operation(typeid(decltype(*this)), { index }, 1, {}, position, loc) {}

    auto Result() const { return Results()[0]; }
};


struct SampleOp : Operation {
    SampleOp(Value tensor, Value index, std::optional<dag::Location> loc = {})
        : Operation(typeid(decltype(*this)), { tensor, index }, 1, {}, {}, loc) {}

    auto Result() const { return Results()[0]; }
};


} // namespace dag