function(get_python_include_dir _PYTHON_IN_PATH OUTPUT_VARIABLE)
    execute_process(
        COMMAND ${_PYTHON_IN_PATH} -c "import sysconfig; print(sysconfig.get_path('include'))"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_VARIABLE OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(${OUTPUT_VARIABLE} ${OUTPUT} PARENT_SCOPE)
endfunction()