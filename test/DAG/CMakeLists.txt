add_executable(TestDAG)

target_sources(TestDAG
    PRIVATE
        main.cpp
        Test_Function.cpp
)

target_link_libraries(TestDAG LLVMFileCheck LLVMSupport)
target_link_libraries(TestDAG DAG)
target_link_libraries(TestDAG TestTools)

find_package(Catch2)
target_link_libraries(TestDAG Catch2::Catch2)