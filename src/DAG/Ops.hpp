#include "Operation.hpp"


namespace dag {


template <class ConcreteOp>
struct Op {
    Op(std::vector<Value> operands,
       size_t numResults,
       std::vector<Region> regions,
       std::any attributes,
       std::optional<Location> loc = {})
        : operation(static_cast<ConcreteOp&>(*this), std::move(operands), numResults, std::move(regions), attributes, loc) {}
    Operation operation;
    operator Operation() const { return operation; }
    operator std::shared_ptr<OperationImpl>() const { return operation; }


    std::type_index Type() const { return operation.Type(); }
    std::span<Operand> Operands() const { return operation.Operands(); }
    std::span<Value> Results() const { return operation.Results(); }
    std::span<Region> Regions() const { return operation.Regions(); }
    const std::any& Attributes() const { return operation.Attributes(); }
    const std::optional<Location>& Location() const { return operation.Location(); }
};


struct FuncAttr {
    std::string name;
    std::shared_ptr<ast::FunctionType> signature;
    bool isPublic;
};


struct ModuleOp : Op<ModuleOp> {
    ModuleOp(std::optional<dag::Location> loc = {})
        : Op({}, 0, { Region{} }, {}, loc) {}

    Region& Body() {
        return operation.Regions().front();
    }
    template <class OpT, class... Args>
    OpT Create(Args&&... args) {
        auto op = OpT(std::forward<Args>(args)...);
        Body().operations.push_back(op);
        return op;
    }
};


struct FuncOp : Op<FuncOp> {
    FuncOp(std::string name,
           std::shared_ptr<ast::FunctionType> signature,
           bool isPublic = true,
           std::optional<dag::Location> loc = {})
        : Op({}, 0, { Region{} }, FuncAttr{ name, signature, isPublic }, loc) {
        size_t index = 0;
        for (auto type : signature->parameters) {
            Body().args.push_back(Value(*this, index++));
        }
    }

    Region& Body() {
        return operation.Regions().front();
    }
};


struct ReturnOp : Op<ReturnOp> {
    ReturnOp(std::vector<Value> values, std::optional<dag::Location> loc = {})
        : Op(values, 0, {}, {}, loc) {}
};


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


struct ArithmeticOp : Op<ArithmeticOp> {
    ArithmeticOp(Value lhs, Value rhs, eArithmeticFunction function, std::optional<dag::Location> loc = {})
        : Op({ lhs, rhs }, 1, {}, function, loc),
          lhs(lhs),
          rhs(rhs),
          result(Op::Results()[0]) {
    }
    Value lhs;
    Value rhs;
    Value result;
};


} // namespace dag