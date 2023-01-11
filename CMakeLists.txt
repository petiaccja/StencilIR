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

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
link_libraries(Threads::Threads)

# LLVM dependency
find_package(MLIR QUIET)
if(NOT MLIR_FOUND)
	message("Installing LLVM from sources... this might take a while.")
	message("You can also set the MLIR_DIR and LLVM_DIR environment variables and reconfigure.")

	set(EXTERNAL_LLVM_SOURCE_DIR "${CMAKE_SOURCE_DIR}/externals/LLVM")
	set(EXTERNAL_LLVM_BINARY_DIR "${CMAKE_BINARY_DIR}/externals/LLVM")
	file(MAKE_DIRECTORY "${EXTERNAL_LLVM_BINARY_DIR}")

	execute_process(
		COMMAND "${CMAKE_COMMAND}"
			-S "${EXTERNAL_LLVM_SOURCE_DIR}"
			-B "${EXTERNAL_LLVM_BINARY_DIR}"
			"-DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}"
			"-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
			"-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
			"-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
			"-G" ${CMAKE_GENERATOR}
		WORKING_DIRECTORY "${EXTERNAL_LLVM_BINARY_DIR}"
		COMMAND_ERROR_IS_FATAL ANY
	)
	execute_process(
		COMMAND ${CMAKE_COMMAND} --build "${EXTERNAL_LLVM_BINARY_DIR}" --parallel
		WORKING_DIRECTORY "${EXTERNAL_LLVM_BINARY_DIR}"
		COMMAND_ERROR_IS_FATAL ANY
	)
	
	set(MLIR_DIR "${EXTERNAL_LLVM_BINARY_DIR}/install/lib/cmake/mlir")
	set(LLVM_DIR "${EXTERNAL_LLVM_BINARY_DIR}/install/lib/cmake/llvm")
	find_package(MLIR REQUIRED CONFIG)
endif()


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
add_subdirectory(python/src)

# Distribution
set_target_properties(stencilir PROPERTIES INSTALL_RPATH "$ORIGIN")
install(
	TARGETS stencilir
	RUNTIME_DEPENDENCIES
	DESTINATION "${CMAKE_BINARY_DIR}/install/python"
)