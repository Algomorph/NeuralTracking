# Option 1: Do not define "PYTHON_EXECUTABLE", but run `cmake ..` within your
#           virtual environment. CMake will pick up the python executable in the
#           virtual environment.
# Option 2: You can also define `cmake -DPYTHON_EXECUTABLE` to specify a python
#           executable.

include(get_python_version)

# Check for both 'python' and 'python3' in path, if both exist choose default
# based on the highest version
find_program(PYTHON_IN_PATH "python")
find_program(PYTHON3_IN_PATH "python3")

if(PYTHON_IN_PATH)
    get_python_version(${PYTHON_IN_PATH} PYTHON_IN_PATH_VERSION)
endif()
if(PYTHON3_IN_PATH)
    get_python_version(${PYTHON3_IN_PATH} PYTHON3_IN_PATH_VERSION)
endif()

if(PYTHON_IN_PATH_VERSION AND PYTHON3_IN_PATH_VERSION)
    if(PYTHON3_IN_PATH_VERSION VERSION_GREATER PYTHON_IN_PATH_VERSION)

        set(PYTHON_IN_PATH ${PYTHON3_IN_PATH})
    endif()
elseif(PYTHON3_VERSION)
    set(PYTHON_IN_PATH ${PYTHON3_IN_PATH})
endif()

if (NOT PYTHON_EXECUTABLE)
    set(PYTHON_EXECUTABLE ${PYTHON_IN_PATH} CACHE FILEPATH "Python exectuable to use")
    message(STATUS "Using python from PATH: ${PYTHON_EXECUTABLE}")
else()
    message(STATUS "Using python from PYTHON_EXECUTABLE variable: ${PYTHON_EXECUTABLE}")
    if ("${PYTHON_EXECUTABLE}" STREQUAL "${PYTHON_IN_PATH}")
        message(STATUS "(PYTHON_EXECUTABLE matches python from PATH)")
    else()
        message(STATUS "(PYTHON_EXECUTABLE does NOT match python from PATH: ${PYTHON_IN_PATH})")
    endif()
endif()
