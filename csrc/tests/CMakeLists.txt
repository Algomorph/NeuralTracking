function(add_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PUBLIC nnrt_cpp test_utils)
    target_link_libraries(${name} PUBLIC Python3::Python ${CATCH2_TARGET})
endfunction()

if (BUILD_CPP_TESTS)
    ### test utilities library ###
    # since dlib is used only for the tests, only look for dependency here
    find_package(dlib REQUIRED)

    set(STATIC_TEST_DATA_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test_data/)
    set(GENERATED_TEST_DATA_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test_data/)
    file(MAKE_DIRECTORY ${GENERATED_TEST_DATA_DIRECTORY})
    configure_file(test_utils/test_data_paths.hpp.in test_utils/test_data_paths.hpp @ONLY)

    add_library(test_utils STATIC
        test_utils/test_utils.hpp
        test_utils/test_utils.tpp
        test_utils/test_utils.cpp
        test_utils/compile_time_string_concatenation.hpp
    )
    target_link_libraries(test_utils PUBLIC dlib::dlib Python3::Module)
    target_include_directories(test_utils PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/test_utils)


    ### add tests ###
    add_test(test_graph_proc test_graph_proc.cpp)
    add_test(test_index_conversions test_index_conversions.cpp)
    add_test(test_array_comparisons test_array_comparisons.cpp)


endif ()