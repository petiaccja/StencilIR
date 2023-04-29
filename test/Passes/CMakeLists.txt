add_executable(TestPasses)

target_sources(TestPasses
    PRIVATE
        main.cpp
        Test_Passes.cpp
)

target_link_libraries(TestPasses LLVMFileCheck LLVMSupport)
target_link_libraries(TestPasses StencilDialect Conversion Transforms)
target_link_libraries(TestPasses TestTools)

find_package(Catch2)
target_link_libraries(TestPasses Catch2::Catch2)

target_compile_definitions(TestPasses PRIVATE FILE_CHECK_DIR="${CMAKE_CURRENT_LIST_DIR}")