#include "Operation.hpp"


namespace dag {



Value::Value(Operation def, size_t index)
    : impl(std::make_shared<ValueImpl>(ValueImpl{ std::weak_ptr{ std::shared_ptr<OperationImpl>{ def } }, {}, index })) {}

Operation Value::DefiningOp() const {
    auto locked = impl->def.lock();
    if (!locked) {
        throw std::logic_error("operation that created this value has been deleted");
    }
    return locked;
}

size_t Value::Index() const {
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


std::type_index Operation::Type() const { return impl->m_type; }
std::span<Operand> Operation::Operands() const { return impl->m_operands; }
std::span<Value> Operation::Results() const { return impl->m_results; }
std::span<Region> Operation::Regions() const { return impl->m_regions; }
const std::any& Operation::Attributes() const { return impl->m_attributes; }
const std::optional<Location>& Operation::Location() const { return impl->m_loc; }


} // namespace dag