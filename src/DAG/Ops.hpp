#include "Operation.hpp"


namespace dag {


template <class ConcreteOp>
struct Op {
    Op(std::span<const Operand> operands,
       std::span<const Result> results,
       std::span<const Region> regions,
       std::any attributes,
       std::optional<Location> loc = {})
        : operation(static_cast<ConcreteOp&>(*this), operands, results, regions, loc) {}
    Operation operation;
    operator Operation() const { return operation; }
    operator std::shared_ptr<OperationImpl>() const { return operation; }
    std::span<const std::shared_ptr<ResultImpl>> Results() { return operation->Results(); }
};


struct FuncAttr {
    std::string name;
    std::shared_ptr<ast::FunctionType> signature;
    bool isPublic;
};


inline Region CreateFunctionRegion(Operation op, std::shared_ptr<ast::FunctionType> signature) {
    std::vector<Result> args;
    size_t index = 0;
    for (auto type : signature->results) {
        args.push_back(Result(op, index++));
    }
    return Region{ args, {} };
}


inline std::vector<Operand> CreateOperands(Operation op, std::span<Result> values) {
    std::vector<Operand> operands;
    std::ranges::transform(values, std::back_inserter(operands), [&](const Result& result) {
        return Operand(result, op);
    });
    return operands;
}


struct ModuleOp : Op<ModuleOp> {
    ModuleOp(std::optional<Location> loc = {})
        : Op({},
             {},
             std::array{ Region{} },
             {},
             loc) {}

    RegionImpl& Body() {
        return operation->Regions().front();
    }
};


struct FuncOp : Op<FuncOp> {
    FuncOp(std::string name,
           std::shared_ptr<ast::FunctionType> signature,
           bool isPublic = true,
           std::optional<Location> loc = {})
        : Op({},
             {},
             std::array{ CreateFunctionRegion(*this, signature) },
             FuncAttr{ name, signature, isPublic },
             loc) {}

    RegionImpl& Body() {
        return operation->Regions().front();
    }
};


struct ReturnOp : Op<ReturnOp> {
    ReturnOp(std::vector<Result> values, std::optional<Location> loc = {})
        : Op(CreateOperands(*this, values), {}, {}, {}, loc) {}
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
    ArithmeticOp(Result lhs, Result rhs, eArithmeticFunction function, std::optional<Location> loc = {})
        : Op(std::array{ Operand(lhs, *this), Operand(lhs, *this) },
             std::array{ Result(*this, 0) },
             {},
             function,
             loc) {}
};


} // namespace dag