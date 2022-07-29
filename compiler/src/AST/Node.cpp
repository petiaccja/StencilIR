#include "Node.hpp"

namespace ast {

std::optional<Location> Node::GetLocation() const {
    return m_location;
}

void Node::SetLocation(Location location) {
    m_location = location;
}

void Node::ClearLocation() {
    m_location = {};
}


void Node::AddChild(std::shared_ptr<Node> child) {
    m_children.push_back(std::move(child));
}

void Node::RemoveChild(std::shared_ptr<Node> child) {
    std::erase(m_children, child);
}

const std::vector<std::shared_ptr<Node>>& Node::GetChildren() const {
    return m_children;
}

} // namespace ast