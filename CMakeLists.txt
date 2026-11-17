cmake_minimum_required(VERSION 3.25)
project(StringKernel4LIBSVM VERSION 1.1.0 LANGUAGES CXX)

include(CTest)
include(GNUInstallDirs)

option(STRINGKERNEL_BUILD_CLI "Build converter and official LIBSVM command-line tools" ON)
option(STRINGKERNEL_BUILD_PYTHON "Build the Python extension" OFF)

option(STRINGKERNEL_SANITIZERS "Enable address and undefined-behavior sanitizers" OFF)
option(STRINGKERNEL_COVERAGE "Enable GCC coverage for project sources" OFF)
include(cmake/Cpu.cmake)
find_package(Threads REQUIRED)

if(STRINGKERNEL_SANITIZERS AND (WIN32 OR NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"))
  message(FATAL_ERROR "Sanitizers require GCC or Clang on a supported Unix platform")
endif()
if(STRINGKERNEL_COVERAGE AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  message(FATAL_ERROR "Coverage requires GCC")
endif()
if(STRINGKERNEL_SANITIZERS AND STRINGKERNEL_COVERAGE)
  message(FATAL_ERROR "Use separate builds for sanitizers and coverage")
endif()

function(stringkernel_project_target target)
  target_compile_features(${target} PRIVATE cxx_std_17)
  set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /WX /permissive- /utf-8)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
  if(STRINGKERNEL_SANITIZERS)
    target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${target} PUBLIC -fsanitize=address,undefined)
  endif()
  if(STRINGKERNEL_COVERAGE)
    target_compile_options(${target} PRIVATE --coverage -fprofile-update=atomic -O0 -g)
    target_link_options(${target} PUBLIC --coverage)
  endif()
endfunction()

if(STRINGKERNEL_BUILD_PYTHON)
  include(cmake/Python.cmake)
  if(NOT STRINGKERNEL_BUILD_CLI AND NOT BUILD_TESTING)
    # A wheel needs mathematics only, not CLI parsing or filesystem dependencies.
    return()
  endif()
endif()

add_library(stringkernel STATIC src/kernels.cpp src/kernel_evaluator.cpp src/dataset.cpp src/libsvm_writer.cpp
  src/execution.cpp src/parallel_rows.cpp)
stringkernel_cpu_sources(stringkernel)
target_link_libraries(stringkernel PUBLIC Threads::Threads)
set_target_properties(stringkernel PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(stringkernel PUBLIC cxx_std_17)
target_include_directories(stringkernel PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
stringkernel_project_target(stringkernel)

if(STRINGKERNEL_BUILD_CLI)
  add_library(stringkernel_options STATIC src/cli_options.cpp)
  target_link_libraries(stringkernel_options PUBLIC stringkernel)
  target_include_directories(stringkernel_options PUBLIC "${PROJECT_SOURCE_DIR}/src")
  stringkernel_project_target(stringkernel_options)

  include(cmake/Libsvm.cmake)
  add_library(libsvm STATIC "${libsvm_dir}/svm.cpp")
  target_include_directories(libsvm PUBLIC "${libsvm_dir}")
  set_target_properties(libsvm PROPERTIES CXX_EXTENSIONS OFF)
  set_source_files_properties("${libsvm_dir}/svm-train.c" "${libsvm_dir}/svm-predict.c"
    PROPERTIES LANGUAGE CXX)
  add_executable(svm-train "${libsvm_dir}/svm-train.c")
  add_executable(svm-predict "${libsvm_dir}/svm-predict.c")
  target_link_libraries(svm-train PRIVATE libsvm)
  target_link_libraries(svm-predict PRIVATE libsvm)
  if(MSVC)
    target_compile_definitions(libsvm PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_definitions(svm-train PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_definitions(svm-predict PRIVATE _CRT_SECURE_NO_WARNINGS)
  endif()

  add_executable(string-kernels src/cli.cpp)
  target_link_libraries(string-kernels PRIVATE stringkernel_options)
  target_include_directories(string-kernels PRIVATE "${libsvm_dir}")
  target_compile_definitions(string-kernels PRIVATE STRINGKERNEL_VERSION="${PROJECT_VERSION}")
  stringkernel_project_target(string-kernels)

  install(TARGETS string-kernels svm-train svm-predict
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  install(FILES "${libsvm_dir}/COPYRIGHT"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/StringKernel4LIBSVM)
endif()

install(TARGETS stringkernel EXPORT stringkernelTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/stringkernel DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT stringkernelTargets NAMESPACE stringkernel::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/StringKernel4LIBSVM)
install(FILES LICENSE THIRD_PARTY_NOTICES.md
  DESTINATION ${CMAKE_INSTALL_DATADIR}/StringKernel4LIBSVM)

if(BUILD_TESTING)
  if(NOT STRINGKERNEL_BUILD_CLI)
    message(FATAL_ERROR "C++ integration tests require STRINGKERNEL_BUILD_CLI=ON")
  endif()
  include(cmake/Dependencies.cmake)
  find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter)
  add_subdirectory(tests)
endif()
