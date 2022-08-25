add_subdirectory(TestGpu)
add_subdirectory(TestStructured)
add_subdirectory(TestUnstructured)

add_executable(test_integration)

target_sources(test_integration
    PRIVATE
        main.cpp
)

find_package(Catch2)
target_link_libraries(test_integration Catch2::Catch2)