add_executable(cuda12trainc "cuda12trainc.cc")
target_include_directories(cuda12trainc PRIVATE "${ONNXPATHCUDA12}/include")
target_link_libraries(cuda12trainc PRIVATE "${ONNXPATHCUDA12}/lib/libonnxruntime.so")
target_link_libraries(cuda12trainc PRIVATE "${ONNXPATHCUDA12}/lib/libonnxruntime_providers_cuda.so")
target_link_libraries(cuda12trainc PRIVATE "${ONNXPATHCUDA12}/lib/libonnxruntime_providers_shared.so")
target_link_libraries(cuda12trainc PRIVATE "${ONNXPATHCUDA12}/lib/libonnxruntime_providers_tensorrt.so")


