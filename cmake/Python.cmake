find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 3.1.0 EXACT CONFIG REQUIRED)
pybind11_add_module(_native python/bindings.cpp src/kernels.cpp src/kernel_evaluator.cpp)
target_include_directories(_native PRIVATE "${PROJECT_SOURCE_DIR}/include" "${PROJECT_SOURCE_DIR}/src")
target_compile_definitions(_native PRIVATE STRINGKERNEL_VERSION="${PROJECT_VERSION}")
stringkernel_project_target(_native)
install(TARGETS _native LIBRARY DESTINATION stringkernels COMPONENT python
  RUNTIME DESTINATION stringkernels COMPONENT python)
install(FILES THIRD_PARTY_NOTICES.md DESTINATION stringkernels COMPONENT python)
