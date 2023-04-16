#include <filesystem>
#include <optional>
#include <string_view>


namespace sir {


std::optional<std::filesystem::path> GetModulePath(std::string_view moduleNameRegex);


}