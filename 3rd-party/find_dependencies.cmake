set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/3rd-party/CMake)

#
# NNRT 3rd party library integration
#
set(NNRT_3RDPARTY_DIR "${PROJECT_SOURCE_DIR}/3rd-party")

# EXTERNAL_MODULES
# CMake modules we depend on in our public interface. These are modules we
# need to find_package() in our CMake config script, because we will use their
# targets.
set(NNRT_3RDPARTY_EXTERNAL_MODULES)

# PUBLIC_TARGETS
# CMake targets we link against in our public interface. They are
# either locally defined and installed, or imported from an external module
# (see above).
set(NNRT_3RDPARTY_PUBLIC_TARGETS)

# HEADER_TARGETS
# CMake targets we use in our public interface, but as a special case we do not
# need to link against the library. This simplifies dependencies where we merely
# expose declared data types from other libraries in our public headers, so it
# would be overkill to require all library users to link against that dependency.
set(NNRT_3RDPARTY_HEADER_TARGETS)

# PRIVATE_TARGETS
# CMake targets for dependencies which are not exposed in the public API. This
# will probably include HEADER_TARGETS, but also anything else we use internally.
set(NNRT_3RDPARTY_PRIVATE_TARGETS)


find_package(PkgConfig QUIET)


#
# import_3rdparty_library(name ...)
#
# Imports a third-party library that has been built independently in a sub project.
#
# Valid options:
#    PUBLIC
#        the library belongs to the public interface and must be installed
#    HEADER
#        the library headers belong to the public interface and will be
#        installed, but the library is linked privately.
#    INCLUDE_DIRS
#        the temporary location where the library headers have been installed.
#        Trailing slashes have the same meaning as with install(DIRECTORY).
#        If your include is "#include <x.hpp>" and the path of the file is
#        "/path/to/libx/x.hpp" then you need to pass "/path/to/libx/"
#        with the trailing "/". If you have "#include <libx/x.hpp>" then you
#        need to pass "/path/to/libx".
#    LIBRARIES
#        the built library name(s). It is assumed that the library is static.
#        If the library is PUBLIC, it will be renamed to NNRT_${name} at
#        install time to prevent name collisions in the install space.
#    LIB_DIR
#        the temporary location of the library. Defaults to
#        CMAKE_ARCHIVE_OUTPUT_DIRECTORY.
#
function(import_3rdparty_library name)
    cmake_parse_arguments(arg "PUBLIC;HEADER" "LIB_DIR" "INCLUDE_DIRS;LIBRARIES" ${ARGN})
    if(arg_UNPARSED_ARGUMENTS)
        message(STATUS "Unparsed: ${arg_UNPARSED_ARGUMENTS}")
        message(FATAL_ERROR "Invalid syntax: import_3rdparty_library(${name} ${ARGN})")
    endif()
    if(NOT arg_LIB_DIR)
        set(arg_LIB_DIR "${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
    endif()
    add_library(${name} INTERFACE)
    if(arg_INCLUDE_DIRS)
        foreach(incl IN LISTS arg_INCLUDE_DIRS)
            if (incl MATCHES "(.*)/$")
                set(incl_path ${CMAKE_MATCH_1})
            else()
                get_filename_component(incl_path "${incl}" DIRECTORY)
            endif()
            target_include_directories(${name} SYSTEM INTERFACE $<BUILD_INTERFACE:${incl_path}>)
            if(arg_PUBLIC OR arg_HEADER)
                install(DIRECTORY ${incl} DESTINATION ${NNRT_INSTALL_INCLUDE_DIR}/open3d/3rdparty
                    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
                    )
                target_include_directories(${name} INTERFACE $<INSTALL_INTERFACE:${NNRT_INSTALL_INCLUDE_DIR}/open3d/3rdparty>)
            endif()
        endforeach()
    endif()
    if(arg_LIBRARIES)
        list(LENGTH arg_LIBRARIES libcount)
        foreach(arg_LIBRARY IN LISTS arg_LIBRARIES)
            set(library_filename ${CMAKE_STATIC_LIBRARY_PREFIX}${arg_LIBRARY}${CMAKE_STATIC_LIBRARY_SUFFIX})
            if(libcount EQUAL 1)
                set(installed_library_filename ${CMAKE_STATIC_LIBRARY_PREFIX}${PROJECT_NAME}_${name}${CMAKE_STATIC_LIBRARY_SUFFIX})
            else()
                set(installed_library_filename ${CMAKE_STATIC_LIBRARY_PREFIX}${PROJECT_NAME}_${name}_${arg_LIBRARY}${CMAKE_STATIC_LIBRARY_SUFFIX})
            endif()
            target_link_libraries(${name} INTERFACE $<BUILD_INTERFACE:${arg_LIB_DIR}/${library_filename}>)
            if(NOT BUILD_SHARED_LIBS OR arg_PUBLIC)
                install(FILES ${arg_LIB_DIR}/${library_filename}
                    DESTINATION ${NNRT_INSTALL_LIB_DIR}
                    RENAME ${installed_library_filename}
                    )
                target_link_libraries(${name} INTERFACE $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${NNRT_INSTALL_LIB_DIR}/${installed_library_filename}>)
            endif()
        endforeach()
    endif()
    if(NOT BUILD_SHARED_LIBS OR arg_PUBLIC)
        install(TARGETS ${name} EXPORT ${PROJECT_NAME}Targets)
    endif()
    add_library(${PROJECT_NAME}::${name} ALIAS ${name})
endfunction()

#
# build_3rdparty_library(name ...)
#
# Builds a third-party library from source
#
# Valid options:
#    PUBLIC
#        the library belongs to the public interface and must be installed
#    HEADER
#        the library headers belong to the public interface, but the library
#        itself is linked privately
#    INCLUDE_ALL
#        install all files in the include directories. Default is *.h, *.hpp
#    DIRECTORY <dir>
#        the library sources are in the subdirectory <dir> of 3rdparty/
#    INCLUDE_DIRS <dir> [<dir> ...]
#        include headers are in the subdirectories <dir>. Trailing slashes
#        have the same meaning as with install(DIRECTORY). <dir> must be
#        relative to the library source directory.
#        If your include is "#include <x.hpp>" and the path of the file is
#        "path/to/libx/x.hpp" then you need to pass "path/to/libx/"
#        with the trailing "/". If you have "#include <libx/x.hpp>" then you
#        need to pass "path/to/libx".
#    SOURCES <src> [<src> ...]
#        the library sources. Can be omitted for header-only libraries.
#        All sources must be relative to the library source directory.
#    LIBS <target> [<target> ...]
#        extra link dependencies
#
function(build_3rdparty_library name)
    cmake_parse_arguments(arg "PUBLIC;HEADER;INCLUDE_ALL" "DIRECTORY" "INCLUDE_DIRS;SOURCES;LIBS" ${ARGN})
    if(arg_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "Invalid syntax: build_3rdparty_library(${name} ${ARGN})")
    endif()
    if(NOT arg_DIRECTORY)
        set(arg_DIRECTORY "${name}")
    endif()
    if(arg_INCLUDE_DIRS)
        set(include_dirs)
        foreach(incl IN LISTS arg_INCLUDE_DIRS)
            list(APPEND include_dirs "${NNRT_3RDPARTY_DIR}/${arg_DIRECTORY}/${incl}")
        endforeach()
    else()
        set(include_dirs "${NNRT_3RDPARTY_DIR}/${arg_DIRECTORY}/")
    endif()
    message(STATUS "Building library ${name} from source")
    if(arg_SOURCES)
        set(sources)
        foreach(src ${arg_SOURCES})
            list(APPEND sources "${NNRT_3RDPARTY_DIR}/${arg_DIRECTORY}/${src}")
        endforeach()
        add_library(${name} STATIC ${sources})
        foreach(incl IN LISTS include_dirs)
            if (incl MATCHES "(.*)/$")
                set(incl_path ${CMAKE_MATCH_1})
            else()
                get_filename_component(incl_path "${incl}" DIRECTORY)
            endif()
            target_include_directories(${name} SYSTEM PUBLIC
                $<BUILD_INTERFACE:${incl_path}>
                )
        endforeach()
        target_include_directories(${name} PUBLIC
            $<INSTALL_INTERFACE:${NNRT_INSTALL_INCLUDE_DIR}/open3d/3rdparty>
            )
        open3d_set_global_properties(${name})
        set_target_properties(${name} PROPERTIES
            OUTPUT_NAME "${PROJECT_NAME}_${name}"
            )
        if(arg_LIBS)
            target_link_libraries(${name} PRIVATE ${arg_LIBS})
        endif()
    else()
        add_library(${name} INTERFACE)
        foreach(incl IN LISTS include_dirs)
            if (incl MATCHES "(.*)/$")
                set(incl_path ${CMAKE_MATCH_1})
            else()
                get_filename_component(incl_path "${incl}" DIRECTORY)
            endif()
            target_include_directories(${name} SYSTEM INTERFACE
                $<BUILD_INTERFACE:${incl_path}>
                )
        endforeach()
        target_include_directories(${name} INTERFACE
            $<INSTALL_INTERFACE:${NNRT_INSTALL_INCLUDE_DIR}/open3d/3rdparty>
            )
    endif()
    if(NOT BUILD_SHARED_LIBS OR arg_PUBLIC)
        install(TARGETS ${name} EXPORT ${PROJECT_NAME}Targets
            RUNTIME DESTINATION ${NNRT_INSTALL_BIN_DIR}
            ARCHIVE DESTINATION ${NNRT_INSTALL_LIB_DIR}
            LIBRARY DESTINATION ${NNRT_INSTALL_LIB_DIR}
            )
    endif()
    if(arg_PUBLIC OR arg_HEADER)
        foreach(incl IN LISTS include_dirs)
            if(arg_INCLUDE_ALL)
                install(DIRECTORY ${incl}
                    DESTINATION ${NNRT_INSTALL_INCLUDE_DIR}/open3d/3rdparty
                    )
            else()
                install(DIRECTORY ${incl}
                    DESTINATION ${NNRT_INSTALL_INCLUDE_DIR}/open3d/3rdparty
                    FILES_MATCHING
                    PATTERN "*.h"
                    PATTERN "*.hpp"
                    )
            endif()
        endforeach()
    endif()
    add_library(${PROJECT_NAME}::${name} ALIAS ${name})
endfunction()
# Convenience function to link against all third-party libraries
# We need this because we create a lot of object libraries to assemble
# the main library
function(nnrt_link_3rdparty_libraries target)
    target_link_libraries(${target} PRIVATE ${NNRT_3RDPARTY_PRIVATE_TARGETS})
    target_link_libraries(${target} PUBLIC ${NNRT_3RDPARTY_PUBLIC_TARGETS})
    foreach(dep IN LISTS NNRT_3RDPARTY_HEADER_TARGETS)
        if(TARGET ${dep})
            get_property(inc TARGET ${dep} PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
            if(inc)
                set_property(TARGET ${target} APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${inc})
            endif()
            get_property(inc TARGET ${dep} PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES)
            if(inc)
                set_property(TARGET ${target} APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES ${inc})
            endif()
            get_property(def TARGET ${dep} PROPERTY INTERFACE_COMPILE_DEFINITIONS)
            if(def)
                set_property(TARGET ${target} APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS ${def})
            endif()
        endif()
    endforeach()
endfunction()

# Python
find_package(PythonExecutable REQUIRED) # invokes the module in 3rdparty/CMake

# Eigen3
if(USE_SYSTEM_EIGEN3)
    find_package(Eigen3)
    if(TARGET Eigen3::Eigen)
        message(STATUS "Using installed third-party library Eigen3 ${EIGEN3_VERSION_STRING}")
        # Eigen3 is a publicly visible dependency, so add it to the list of
        # modules we need to find in the NNRT config script.
        list(APPEND NNRT_3RDPARTY_EXTERNAL_MODULES "Eigen3")
        set(EIGEN3_TARGET "Eigen3::Eigen")
    else()
        message(STATUS "Unable to find installed third-party library Eigen3")
        set(USE_SYSTEM_EIGEN3 OFF)
    endif()
endif()
if(NOT USE_SYSTEM_EIGEN3)
    build_3rdparty_library(3rdparty_eigen3 PUBLIC DIRECTORY Eigen INCLUDE_DIRS Eigen INCLUDE_ALL)
    set(EIGEN3_TARGET "3rdparty_eigen3")
endif()
list(APPEND NNRT_3RDPARTY_PUBLIC_TARGETS "${EIGEN3_TARGET}")

# Pybind11
if(USE_SYSTEM_PYBIND11)
    find_package(pybind11)
endif()
if (NOT USE_SYSTEM_PYBIND11 OR NOT TARGET pybind11::module)
    set(USE_SYSTEM_PYBIND11 OFF)
    add_subdirectory(${NNRT_3RDPARTY_DIR}/pybind11)
endif()
if(TARGET pybind11::module)
    set(PYBIND11_TARGET "pybind11::module")
endif()
list(APPEND NNRT_3RDPARTY_PUBLIC_TARGETS "${PYBIND11_TARGET}")

# Pytorch

#find_package(Pytorch REQUIRED)
#list(APPEND NNRT_3RDPARTY_PUBLIC_TARGETS torch)

# Threads
find_package(Threads REQUIRED)

# Catch2
if(USE_SYSTEM_CATCH2)
    find_package(Catch2)
    if(TARGET Catch2::Catch2)
        message(STATUS "Using installed third-party library Catch2")
        list(APPEND NNRT_3RDPARTY_EXTERNAL_MODULES "Catch2")
        set(CATCH2_TARGET "Catch2::Catch2")
    else()
        message(STATUS "Unable to find installed third-party library Catch2")
        set(USE_SYSTEM_CATCH2 OFF)
    endif()
    list(APPEND NNRT_3RDPARTY_PUBLIC_TARGETS ${3rdparty_Catch2})
else()
    include(${NNRT_3RDPARTY_DIR}/Catch2/Catch2.cmake)
    import_3rdparty_library(3rdparty_Catch2
        INCLUDE_DIRS ${CATCH2_INCLUDE_DIRS}
        LIB_DIR      ${CATCH2_LIB_DIR}
        LIBRARIES    ${CATCH2_LIBRARIES}
    )
    set(CATCH2_TARGET "3rdparty_Catch2")
    add_dependencies(3rdparty_Catch2 ext_Catch2)
endif()

message(STATUS "Catch libraries: ${TEST}")

list(APPEND NNRT_3RDPARTY_PUBLIC_TARGETS ${CATCH2_TARGET})

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)