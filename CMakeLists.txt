cmake_minimum_required(VERSION 3.18)

option(DEVELOPER_BUILD      "Add +commit_hash to the project version number" ON )
option(USE_SYSTEM_EIGEN3          "Use system pre-installed eigen3"          OFF)
option(USE_SYSTEM_PYBIND11        "Use system pre-installed pybind11"        OFF)
option(USE_SYSTEM_CATCH2          "Use system pre-installed Catch2"          OFF)
option(GLIBCXX_USE_CXX11_ABI      "Set -D_GLIBCXX_USE_CXX11_ABI=1"           ON)
option(BUILD_CPP_TESTS            "Build C++ tests"                         OFF)

# Parse NNRT version number
file(STRINGS "csrc/version.txt" NNRT_VERSION_READ)
foreach(ver ${NNRT_VERSION_READ})
    if (ver MATCHES "NNRT_VERSION_(MAJOR|MINOR|PATCH) +([^ ]+)$")
        set(NNRT_VERSION_${CMAKE_MATCH_1} "${CMAKE_MATCH_2}" CACHE INTERNAL "")
    endif()
endforeach()

set(NNRT_VERSION_DEVHASH "")

if(DEVELOPER_BUILD)
    execute_process(COMMAND git -C "${CMAKE_SOURCE_DIR}" log --pretty=format:%h -n 1
                    OUTPUT_VARIABLE GIT_REV)
    if (GIT_REV)
        set(NNRT_VERSION_DEVHASH "+${GIT_REV}")
    endif()
endif()

string(CONCAT NNRT_VERSION
    "${NNRT_VERSION_MAJOR}"
    ".${NNRT_VERSION_MINOR}"
    ".${NNRT_VERSION_PATCH}"
)
set(NNRT_VERSION_FULL "${NNRT_VERSION}${NNRT_VERSION_DEVHASH}")


### Set installation paths
if(UNIX OR CYGWIN)
    include(GNUInstallDirs)
    set(NNRT_INSTALL_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}")
    set(NNRT_INSTALL_BIN_DIR "${CMAKE_INSTALL_BINDIR}")
    set(NNRT_INSTALL_LIB_DIR "${CMAKE_INSTALL_LIBDIR}")
    set(NNRT_INSTALL_CMAKE_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
else()
    set(NNRT_INSTALL_INCLUDE_DIR include)
    set(NNRT_INSTALL_BIN_DIR bin)
    set(NNRT_INSTALL_LIB_DIR lib)
    set(NNRT_INSTALL_CMAKE_DIR CMake)
endif()

### project definition, build configuration

project(NNRT VERSION ${NNRT_VERSION} LANGUAGES C CXX)
message(STATUS "NNRT ${NNRT_VERSION_FULL}")

### global functions

function(nnrt_set_global_properties target)
    if(GLIBCXX_USE_CXX11_ABI)
        target_compile_definitions(${target} PUBLIC _GLIBCXX_USE_CXX11_ABI=1)
    else()
        target_compile_definitions(${target} PUBLIC _GLIBCXX_USE_CXX11_ABI=0)
    endif()
endfunction()

### includes for dependency processing

include(3rd-party/find_dependencies.cmake)

### subdirectories

add_subdirectory(csrc)

