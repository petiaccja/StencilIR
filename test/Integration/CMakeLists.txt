add_executable(TestIntegration)

target_sources(TestIntegration
    PRIVATE
        main.cpp
        Utility/RunModule.hpp
)

target_sources(TestIntegration
    PRIVATE
        Test_Optimization.cpp
        Test_ObjectFile.cpp
)

target_link_libraries(TestIntegration AST Compiler Execution IR)

find_package(Catch2)
target_link_libraries(TestIntegration Catch2::Catch2)

if (WIN32)
	set(build_target TestIntegration)
	add_custom_command(
		TARGET ${build_target} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
		$<TARGET_RUNTIME_DLLS:${build_target}> $<TARGET_FILE_DIR:${build_target}>
		COMMAND_EXPAND_LISTS
	)
endif()