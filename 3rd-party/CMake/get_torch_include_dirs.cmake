function(get_torch_include_dirs _PYTHON_IN_PATH OUTPUT_VARIABLE)
    execute_process(
        COMMAND ${_PYTHON_IN_PATH} -c "import torch.utils.cpp_extension as tuc; print(';'.join(tuc.include_paths()))"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_VARIABLE OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(${OUTPUT_VARIABLE} ${OUTPUT} PARENT_SCOPE)
endfunction()