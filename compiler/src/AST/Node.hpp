#pragma once

#include <memory>
#include <optional>
#include <vector>


namespace ast {

struct Location {
    std::string file;
    int line;
    int col;
};

class Node {
public:
    virtual ~Node() = default;

    std::optional<Location> GetLocation() const;
    void SetLocation(Location location);
    void ClearLocation();

    void AddChild(std::shared_ptr<Node> child);
    void RemoveChild(std::shared_ptr<Node> child);
    const std::vector<std::shared_ptr<Node>>& GetChildren() const;

private:
    std::vector<std::shared_ptr<Node>> m_children;
    std::optional<Location> m_location;
};


class Module : public Node {
};

} // namespace ast