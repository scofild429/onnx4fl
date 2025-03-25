#include "iostream"
#include <fstream>
#include <chrono>
#include "onnxruntime_cxx_api.h"

int main() {
  auto start = std::chrono::high_resolution_clock::now();
  printf("You are doing ondevice training in C with CUDA, all codes are commented!\n");
  // const auto& api = Ort::GetApi();
  // struct CudaMemoryDeleter {
  //   Ort::Allocator* alloc_;
    
  //   explicit CudaMemoryDeleter(Ort::Allocator* alloc) {
  //     alloc_ = alloc;
  //   }

  //   void operator()(void* ptr) const {
  //     alloc_->Free(ptr);
  //   }
  // };

  // OrtCUDAProviderOptionsV2* cuda_options = nullptr;
  // api.CreateCUDAProviderOptions(&cuda_options);
  // std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(api.ReleaseCUDAProviderOptions)> rel_cuda_options(cuda_options, api.ReleaseCUDAProviderOptions);
  // std::vector<const char*> keys{"enable_cuda_graph"};
  // std::vector<const char*> values{"1"};
  // api.UpdateCUDAProviderOptions(rel_cuda_options.get(), keys.data(), values.data(), 1);

  // Ort::SessionOptions session_options;
  // api.SessionOptionsAppendExecutionProvider_CUDA_V2(static_cast<OrtSessionOptions*>(session_options), rel_cuda_options.get());


  // // Pass gpu_graph_id to RunOptions through RunConfigs
  // Ort::RunOptions run_option;
  // // gpu_graph_id is optional if the session uses only one cuda graph
  // run_option.AddConfigEntry("gpu_graph_id", "1");

  // Ort::Session session(Ort::Env(), ORT_TSTR("~/Downloads/matmul_2.onnx"), session_options);
  // Ort::MemoryInfo info_cuda("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
  // Ort::Allocator cuda_allocator(session, info_cuda);

  // const std::array<int64_t, 2> x_shape = {3, 2};
  // std::array<float, 3 * 2> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  // auto input_data = std::unique_ptr<void, CudaMemoryDeleter>(cuda_allocator.Alloc(x_values.size() * sizeof(float)),
  // 							     CudaMemoryDeleter(&cuda_allocator));
  // cudaMemcpy(input_data.get(),
  // 	     x_values.data(),
  // 	     sizeof(float) * x_values.size(),
  // 	     cudaMemcpyHostToDevice);

  // // Create an OrtValue tensor backed by data on CUDA memory
  // Ort::Value bound_x = Ort::Value::CreateTensor(info_cuda,
  // 						reinterpret_cast<float*>(input_data.get()),
  // 						x_values.size(),
  // 						x_shape.data(),
  // 						x_shape.size());

  // const std::array<int64_t, 2> expected_y_shape = {3, 2};
  // std::array<float, 3 * 2> expected_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};
  // auto output_data = std::unique_ptr<void, CudaMemoryDeleter>(cuda_allocator.Alloc(expected_y.size() * sizeof(float)),
  //                                                             CudaMemoryDeleter(&cuda_allocator));

  // // Create an OrtValue tensor backed by data on CUDA memory
  // Ort::Value bound_y = Ort::Value::CreateTensor(info_cuda,
  // 						reinterpret_cast<float*>(output_data.get()),
  // 						expected_y.size(),
  // 						expected_y_shape.data(),
  // 						expected_y_shape.size());

  // Ort::IoBinding binding(session);
  // binding.BindInput("X", bound_x);
  // binding.BindOutput("Y", bound_y);

  // // One regular run for necessary memory allocation and graph capturing
  // session.Run(run_option, binding);

  // // After capturing, CUDA graph replay happens from this Run onwards
  // session.Run(run_option, binding);

  // // Update input and then replay CUDA graph with the updated input
  // x_values = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
  // cudaMemcpy(input_data.get(),
  // 	     x_values.data(),
  // 	     sizeof(float) * x_values.size(),
  // 	     cudaMemcpyHostToDevice);
  // session.Run(run_option, binding);

  // Ort::MemoryInfo memory_info("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);  
  // Ort::Env env;
  // Ort::SessionOptions session_options;
  // Ort::Session session(env, ORT_TSTR("~/Downloads/matmul_2.onnx"), session_options);
  // Ort::IoBinding io_binding{session};
  // auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
  // io_binding.BindInput("input1", input_tensor);
  // Ort::MemoryInfo output_mem_info{"Cuda", OrtDeviceAllocator, 0,
  //                                 OrtMemTypeDefault};

  // io_binding.BindOutput("output1", output_mem_info);
  // session.Run(run_options, io_binding);
  
  
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout << "Time taken: " << duration/1000000 << " seconds" << std::endl;

  return 0;
}
