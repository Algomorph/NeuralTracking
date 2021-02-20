# Option 1: Do not define "PYBIND11_STUBGEN_EXECUTABLE", but run `cmake ..` within your
#           virtual environment. CMake will pick up the pybind11-stubgen executable in the
#           virtual environment.
# Option 2: You can also define `cmake -DPYBIND11_STUBGEN_EXECUTABLE` to specify a python
#           executable.

include(get_python_version)

find_program(PYBIND11_STUBGEN_IN_PATH "pybind11-stubgen")

if (NOT PYBIND11_STUBGEN_EXECUTABLE)
    set(PYBIND11_STUBGEN_EXECUTABLE ${PYBIND11_STUBGEN_IN_PATH} CACHE FILEPATH "pybind11-stubgen executable to use")
endif()
