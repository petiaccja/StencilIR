#include "DynamicLinking.hpp"


#if defined(__gnu_linux__)

    #include <dlfcn.h>
    #include <link.h>
    #include <regex>


std::optional<std::filesystem::path> GetModulePath(std::string_view moduleNameRegex) {
    dl_find_object result;
    if (_dl_find_object((void*)&GetModulePath, &result) != 0) {
        return {};
    }
    link_map* map = result.dlfo_link_map;
    while (map) {
        std::filesystem::path modulePath{ map->l_name };
        const std::string fileName = modulePath.filename().string();
        std::regex re{ moduleNameRegex.data() };
        if (std::regex_match(fileName.data(), re)) {
            return modulePath;
        }
        map = map->l_next;
    }
    return {};
}

#elif defined(_WIN32)

    #include <Windows.h>
// clang-format: off
// Windows.h must come first
// clang-format: on
    #include <Psapi.h>
    #include <regex>

std::optional<std::filesystem::path> GetModulePath(std::string_view moduleNameRegex) {
    auto hProcess = GetCurrentProcess();

    std::vector<HMODULE> modules(1024);
    DWORD numModulesLoaded = 0;
    if (EnumProcessModules(hProcess, modules.data(), modules.size(), &numModulesLoaded)) {
        for (DWORD i = 0; i < (numModulesLoaded / sizeof(HMODULE)); i++) {
            std::vector<TCHAR> moduleFileName(MAX_PATH);

            if (GetModuleFileNameEx(hProcess, modules[i], moduleFileName.data(), moduleFileName.size())) {
                std::filesystem::path modulePath{ moduleFileName.data() };
                const std::string fileName = modulePath.filename().string();
                std::regex re{ moduleNameRegex.data() };
                if (std::regex_match(fileName.data(), re)) {
                    return modulePath;
                }
            }
        }
    }

    return {};
}

#elif defined(__APPLE__)

static_assert(false, "TODO: implement this functionality for Mac OS.");

#else

static_assert(false, "Not implemented for this operating system.");

#endif
