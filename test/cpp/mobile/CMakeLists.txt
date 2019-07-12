add_executable(test_export
  test_export.cpp)

target_link_libraries(test_export PRIVATE torch gtest)
target_include_directories(test_export PRIVATE ${ATen_CPU_INCLUDE})
#target_compile_definitions(test_export PRIVATE USE_GTEST)

if (USE_CUDA)
  target_link_libraries(test_export PRIVATE
    ${CUDA_LIBRARIES}
    ${CUDA_NVRTC_LIB}
    ${CUDA_CUDA_LIB}
    ${TORCH_CUDA_LIBRARIES})

  target_compile_definitions(test_export PRIVATE USE_CUDA)
endif()

if (INSTALL_TEST)
  install(TARGETS test_export DESTINATION bin)
  # Install PDB files for MSVC builds
  if (MSVC AND BUILD_SHARED_LIBS)
    install(FILES $<TARGET_PDB_FILE:test_export> DESTINATION bin OPTIONAL)
  endif()
endif()
