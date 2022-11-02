#include <filesystem>
#include <optional>
#include <string_view>


std::optional<std::filesystem::path> GetModulePath(std::string_view moduleNameRegex);