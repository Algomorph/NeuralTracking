function(get_pybind11_include_dir _PYTHON_IN_PATH OUTPUT_VARIABLE)
    execute_process(
        COMMAND ${_PYTHON_IN_PATH} -c "import pybind11; print(pybind11.get_include(False))"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_VARIABLE OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(${OUTPUT_VARIABLE} ${OUTPUT} PARENT_SCOPE)
endfunction()