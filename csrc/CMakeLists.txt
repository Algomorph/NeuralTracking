include(get_python_version)
#include(get_pybind11_include_dir)
include(get_torch_include_dirs)
include(get_python_include_dir)

if (NOT PYTHON_EXECUTABLE)
    message(FATAL_ERROR "PYTHON_EXECUTABLE should be set in top level CMakeLists.txt")
endif()

find_package(Pybind11StubgenExecutable)
if (NOT PYBIND11_STUBGEN_EXECUTABLE)
    message(FATAL_ERROR "PYBIND11_STUBGEN_EXECUTABLE should be set")
endif()

get_python_version(${PYTHON_EXECUTABLE} PYTHON_VERSION)

# We need to get python version to configure some meta files
message(STATUS "Using Python version: ${PYTHON_VERSION}")


set(PIP_PACKAGE_DST_DIR "${CMAKE_BINARY_DIR}/pip_package")
set(STUBS_PACKAGE_DST_DIR "${CMAKE_BINARY_DIR}/stubs")

#file(REMOVE_RECURSE ${PIP_PACKAGE_DST_DIR})
#file(REMOVE_RECURSE ${STUBS_PACKAGE_DST_DIR})
file(MAKE_DIRECTORY ${PIP_PACKAGE_DST_DIR})
file(MAKE_DIRECTORY ${STUBS_PACKAGE_DST_DIR})

pybind11_add_module(nnrt main.cpp
    main.cpp
    cpu/graph_proc.h
    cpu/graph_proc.cpp
    cpu/image_proc.h
    cpu/image_proc.cpp
)

get_torch_include_dirs(${PYTHON_EXECUTABLE} PYTORCH_INCLUDE_DIRS)
get_python_include_dir(${PYTHON_EXECUTABLE} PYTHON_INCLUDE_DIR)

target_include_directories(nnrt PRIVATE
    ${PYTORCH_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PYTHON_INCLUDE_DIR}
)

nnrt_link_3rdparty_libraries(nnrt)

# Use `make pip-package` to create the pip package in the build directory
add_custom_target(pip-package
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/setup.py bdist_wheel --dist-dir ${PIP_PACKAGE_DST_DIR}
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


#add_custom_target(stubs-pip-package
#    COMMAND ${PYTHON_EXECUTABLE} ${STUBS_PACKAGE_DST_DIR}/nnrt-stubs/setup.py bdist_wheel --dist-dir ${STUBS_PACKAGE_DST_DIR}
#    COMMAND echo "pip wheel for stubs created at ${STUBS_PACKAGE_DST_DIR}"
#    WORKING_DIRECTORY ${STUBS_PACKAGE_DST_DIR}
#    DEPENDS python-stubs
#)
#add_custom_target(install-stubs-pip-package
#    COMMAND ${CMAKE_COMMAND}
#    -DPIP_PACKAGE_DST_DIR=${STUBS_PACKAGE_DST_DIR}
#    -P ${CMAKE_CURRENT_SOURCE_DIR}/make_install_pip_package.cmake
#    DEPENDS stubs-pip-package
#)


