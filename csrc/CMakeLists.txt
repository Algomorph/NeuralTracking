### ================== pybind11-stubgen configuration ==================
# Stubgen is currently required for proper code completion in PyCharm and other IDEs

find_package(Pybind11StubgenExecutable)
if (NOT PYBIND11_STUBGEN_EXECUTABLE)
    message(FATAL_ERROR "PYBIND11_STUBGEN_EXECUTABLE should be set (pybind11-stubgen command will be found if present in the path)")
endif()

### ==================  Python configuration ========================
# Required for building the python package and wheel for python pip

include(get_python_version)
include(get_python_include_dir)

get_python_version(${PYTHON_EXECUTABLE} PYTHON_VERSION)
get_python_include_dir(${PYTHON_EXECUTABLE} PYTHON_INCLUDE_DIR)

if (NOT PYTHON_EXECUTABLE)
    message(FATAL_ERROR "PYTHON_EXECUTABLE should be set in top level CMakeLists.txt")
endif()

# We need to get python version to configure some meta files
message(STATUS "Using Python version: ${PYTHON_VERSION}")


set(PIP_PACKAGE_DST_DIR "${CMAKE_BINARY_DIR}/pip_package")
set(STUBS_PACKAGE_DST_DIR "${CMAKE_BINARY_DIR}/stubs")


### ================== Output directory prep ==================

file(MAKE_DIRECTORY ${PIP_PACKAGE_DST_DIR})
file(MAKE_DIRECTORY ${STUBS_PACKAGE_DST_DIR})


# ==================== set up NNRT C++ library ============================
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(HEADER_FILES
    cpu/graph_proc.h
    cpu/image_proc.h
)

set(SOURCE_FILES
    cpu/graph_proc.cpp
    cpu/image_proc.cpp
)

add_library(nnrt_cpp STATIC
    ${HEADER_FILES}
    ${SOURCE_FILES}
)

target_include_directories(nnrt_cpp PUBLIC
    ${PYTHON_INCLUDE_DIR}
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

nnrt_link_3rdparty_libraries(nnrt_cpp)
nnrt_set_global_properties(nnrt_cpp)

# ==================== set up NNRT Python binding library ============================

set(PY_HEADER_FILES
#    ${HEADER_FILES}
)

set(PY_SOURCE_FILES
    main.cpp

#    ${SOURCE_FILES}
)

pybind11_add_module(nnrt main.cpp
    ${PY_HEADER_FILES}
    ${PY_SOURCE_FILES}
)

target_include_directories(nnrt PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)


target_link_libraries(nnrt PUBLIC nnrt_cpp)

# Use `make pip-package` to create the pip package in the build directory
add_custom_target(pip-package
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/setup_pybind_cmake.py bdist_wheel --dist-dir ${PIP_PACKAGE_DST_DIR}
    COMMAND echo "pip wheel created at ${PIP_PACKAGE_DST_DIR}"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS nnrt
)

# Use `make install-pip-package` to install pip wheel package to the current
# python environment.
add_custom_target(install-pip-package
    COMMAND ${CMAKE_COMMAND}
    -DPIP_PACKAGE_DST_DIR=${PIP_PACKAGE_DST_DIR}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/make_install_pip_package.cmake
    DEPENDS pip-package
)

set(PYBIND_STUBGEN_IN_PATH)

add_custom_target(python-stubs
    COMMAND ${PYBIND11_STUBGEN_EXECUTABLE} nnrt -o ${STUBS_PACKAGE_DST_DIR}
    COMMAND echo "pybind11 stubs package created at ${STUBS_PACKAGE_DST_DIR}"
    DEPENDS install-pip-package
)


if(PATH_TO_IDE_STUBS_FILE)
    add_custom_target(python-ide-stubs
        COMMAND mv ${STUBS_PACKAGE_DST_DIR}/nnrt-stubs/__init__.pyi ${PATH_TO_IDE_STUBS_FILE}
        COMMAND echo "pybind11 stubs replaced at ${PATH_TO_IDE_STUBS_FILE}"
        DEPENDS python-stubs
    )
endif()


add_subdirectory(tests)



