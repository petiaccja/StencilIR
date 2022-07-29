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

struct Node {
    virtual ~Node() = default;
    Node(std::optional<Location> loc = {}) : location(loc) {}
    std::optional<Location> location;
};

} // namespace ast