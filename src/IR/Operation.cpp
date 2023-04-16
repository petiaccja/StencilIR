#include "Operation.hpp"


namespace sir {



Value::Value(Operation def, size_t index)
    : impl(std::make_shared<ValueImpl>(ValueImpl{ std::weak_ptr{ std::shared_ptr<OperationImpl>{ def } }, {}, index })) {}

Operation Value::GetOwner() const {
    auto locked = impl->def.lock();
    if (!locked) {
        throw std::logic_error("operation that created this value has been deleted");
    }
    return locked;
}

size_t Value::GetIndex() const {
    return impl->index;
}

void Value::AddUser(Operation user) {
    impl->users.insert(user);
}

void Value::RemoveUser(Operation user) {
    impl->users.erase(user);
}


Operand::Operand(Value source, Operation owner)
    : source(source), owner((std::shared_ptr<OperationImpl>)owner) {
    source.AddUser(owner);
}
Operand::~Operand() {
    source.RemoveUser(owner.lock());
}


Operation::Operation(std::type_index type,
                     std::vector<Value> operands,
                     size_t numResults,
                     std::vector<Region> regions,
                     std::any attributes,
                     std::optional<Location> loc)
    : impl(std::make_shared<OperationImpl>(OperationImpl{ type,
                                                          {},
                                                          {},
                                                          std::move(regions),
                                                          attributes,
                                                          loc })) {
    for (auto& value : operands) {
        impl->operands.push_back(Operand(value, *this));
    }
    for (size_t i = 0; i < numResults; ++i) {
        impl->results.push_back(Value(*this, i));
    }
}

std::type_index Operation::Type() const { return impl->type; }
std::span<Operand> Operation::GetOperands() const { return impl->operands; }
std::span<Value> Operation::GetResults() const { return impl->results; }
std::span<Region> Operation::GetRegions() const { return impl->regions; }
const std::any& Operation::GetAttributes() const { return impl->attributes; }
const std::optional<Location>& Operation::GetLocation() const { return impl->loc; }


} // namespace sir