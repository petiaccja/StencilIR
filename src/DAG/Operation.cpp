#include "Operation.hpp"


namespace dag {


std::vector<std::shared_ptr<ResultImpl>> AsImpls(std::span<const Result> args) {
    std::vector<std::shared_ptr<ResultImpl>> r;
    r.reserve(args.size());
    std::transform(args.begin(), args.end(), std::back_inserter(r), [](const auto& v) {
        return (std::shared_ptr<ResultImpl>)v;
    });
    return r;
}


std::vector<std::shared_ptr<OperandImpl>> AsImpls(std::span<const Operand> args) {
    std::vector<std::shared_ptr<OperandImpl>> r;
    r.reserve(args.size());
    std::transform(args.begin(), args.end(), std::back_inserter(r), [](const auto& v) {
        return (std::shared_ptr<OperandImpl>)v;
    });
    return r;
}


std::vector<RegionImpl> AsImpls(std::span<const Region> args) {
    std::vector<RegionImpl> r;
    r.reserve(args.size());
    std::transform(args.begin(), args.end(), std::back_inserter(r), [](const Region& v) {
        std::vector<std::shared_ptr<ResultImpl>> args;
        std::vector<std::shared_ptr<OperationImpl>> ops;

        args.reserve(v.args.size());
        ops.reserve(v.operations.size());

        std::transform(v.operations.begin(), v.operations.end(), std::back_inserter(ops), [](const Operation& v) {
            return (std::shared_ptr<OperationImpl>)v;
        });
        return RegionImpl{ std::move(args), std::move(ops) };
    });
    return r;
}


Result::Result(Operation def, size_t index)
    : impl(std::make_shared<ResultImpl>((std::shared_ptr<OperationImpl>)def, index)) {}


Operand::Operand(Result result, Operation user)
    : impl(std::make_shared<OperandImpl>((std::shared_ptr<ResultImpl>)result, (std::shared_ptr<OperationImpl>)user)) {}


} // namespace dag