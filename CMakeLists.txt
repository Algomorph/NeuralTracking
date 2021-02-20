cmake_minimum_required(VERSION 3.18)

option(DEVELOPER_BUILD      "Add +commit_hash to the project version number" ON )
option(USE_SYSTEM_EIGEN3          "Use system pre-installed eigen3"          OFF)
option(USE_SYSTEM_PYBIND11        "Use system pre-installed pybind11"        OFF)

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


project(NNRT VERSION ${NNRT_VERSION} LANGUAGES C CXX)
message(STATUS "NNRT ${NNRT_VERSION_FULL}")


include(3rd-party/find_dependencies.cmake)


add_subdirectory(csrc)