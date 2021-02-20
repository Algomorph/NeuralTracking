function(get_python_version _PYTHON_IN_PATH OUTPUT_VARIABLE)
    execute_process(
        COMMAND ${_PYTHON_IN_PATH} -c "import sys; print(str(sys.version_info.major) + \".\" + str(sys.version_info.minor) + \".\" + str(sys.version_info.micro))"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_VARIABLE OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(${OUTPUT_VARIABLE} ${OUTPUT} PARENT_SCOPE)
endfunction()