#include "Operation.hpp"


namespace dag {


template <class ConcreteOp>
struct Op {
    Op(std::vector<Value> operands,
       std::vector<Value> results,
       std::vector<Region> regions,
       std::any attributes,
       std::optional<Location> loc = {})
        : operation(static_cast<ConcreteOp&>(*this), std::move(operands), std::move(results), std::move(regions), attributes, loc) {}
    Operation operation;
    operator Operation() const { return operation; }
    operator std::shared_ptr<OperationImpl>() const { return operation; }
    std::span<Value> Results() { return operation.Results(); }
};


struct FuncAttr {
    std::string name;
    std::shared_ptr<ast::FunctionType> signature;
    bool isPublic;
};


inline Region CreateFunctionRegion(Operation op, std::shared_ptr<ast::FunctionType> signature) {
    std::vector<Value> args;
    size_t index = 0;
    for (auto type : signature->results) {
        args.push_back(Value(op, index++));
    }
    return Region{ args, {} };
}


struct ModuleOp : Op<ModuleOp> {
    ModuleOp(std::optional<Location> loc = {})
        : Op({},
             {},
             { Region{} },
             {},
             loc) {}

    Region& Body() {
        return operation.Regions().front();
    }
};


struct FuncOp : Op<FuncOp> {
    FuncOp(std::string name,
           std::shared_ptr<ast::FunctionType> signature,
           bool isPublic = true,
           std::optional<Location> loc = {})
        : Op({},
             {},
             { CreateFunctionRegion(*this, signature) },
             FuncAttr{ name, signature, isPublic },
             loc) {}

    Region& Body() {
        return operation.Regions().front();
    }
};


struct ReturnOp : Op<ReturnOp> {
    ReturnOp(std::vector<Value> values, std::optional<Location> loc = {})
        : Op(values, {}, {}, {}, loc) {}
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
    ArithmeticOp(Value lhs, Value rhs, eArithmeticFunction function, std::optional<Location> loc = {})
        : Op({ lhs, rhs },
             { Value(*this, 0) },
             {},
             function,
             loc) {}
};


} // namespace dag