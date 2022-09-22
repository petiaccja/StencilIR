project(StencilIR)

cmake_minimum_required(VERSION 3.20.0)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

# Cpp flags
enable_language(CXX)
set(CMAKE_CXX_STANDARD 20)

option(ENABLE_LLVM_COV "Adds compiler flags to generate LLVM source-based code coverage. Only works with Clang." OFF)
option(ENABLE_LLVM_ADDRESS_SANITIZER "Adds compiler flags to generate LLVM source-based code coverage. Only works with Clang." OFF)
option(ENABLE_LLVM_MEMORY_SANITIZER "Adds compiler flags to generate LLVM source-based code coverage. Only works with Clang." OFF)

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
	if (ENABLE_LLVM_COV)
		add_compile_options("-fprofile-instr-generate" "-fcoverage-mapping" "-mllvm" "-enable-name-compression=false")
		add_link_options("-fprofile-instr-generate" "-fcoverage-mapping")
	endif()
	if (ENABLE_LLVM_ADDRESS_SANITIZER)
		add_compile_options("-fsanitize=address")
		add_link_options("-fsanitize=address")
	endif()
	if (ENABLE_LLVM_MEMORY_SANITIZER)
		add_compile_options("-fsanitize=memory")
		add_link_options("-fsanitize=memory")
	endif()
	if ("${CMAKE_CXX_COMPILER_FRONTEND_VARIANT}" STREQUAL "MSVC")
		set(RELWITHDEBINFO_OPTIONS "/Ob2")
		set(RELWITHDEBINFO_REMOVE "/Ob1")
		foreach(option ${RELWITHDEBINFO_REMOVE})
			string(REPLACE ${option} "" CMAKE_CXX_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
		endforeach()
		add_compile_options("$<$<CONFIG:RELWITHDEBINFO>:${RELEASE_OPTIONS}>")
	endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
	add_compile_options("/MP")
	set(RELWITHDEBINFO_OPTIONS "/Ob2")
	set(RELWITHDEBINFO_REMOVE "/Ob1")
	foreach(option ${RELWITHDEBINFO_REMOVE})
		string(REPLACE ${option} "" CMAKE_CXX_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
	endforeach()
	add_compile_options("$<$<CONFIG:RELWITHDEBINFO>:${RELEASE_OPTIONS}>")
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
link_libraries(Threads::Threads)

# Conan library dependencies
include(conan.cmake)
conan_cmake_configure(
	REQUIRES
		pybind11/2.10.0
		catch2/2.13.9
	GENERATORS cmake_find_package
)
conan_cmake_autodetect(conan_settings)
conan_cmake_install(PATH_OR_REFERENCE . BUILD missing SETTINGS ${conan_settings} OUTPUT_QUIET)

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
if (NOT DEFINED CMAKE_LIBRARY_OUTPUT_DIRECTORY)
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Subdirectories
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(python/bindings)