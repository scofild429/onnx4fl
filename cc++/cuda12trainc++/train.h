#ifndef TRAIN_H
#define TRAIN_H

#include "iostream"
#include "session.h"

float train_step(sessions::SessionCache *session_cache, float *batches, std::array<float, TARGET_CLASS_NUMBER> labels) {

  ///////////////////////////////////////////////
  //  Ort::IoBinding io_binding{&session_cache->inference_session};
  
  const std::vector<int64_t> input_shape({
      1,
      INSTANCE_WORDS_LENGTH,
      INSTANCE_WORDS_DIM});

  int64_t label_at[1] = {};
  label_at[0] = std::distance(labels.begin(), std::max_element(labels.begin(), labels.end()));
  const std::vector<int64_t> labels_shape({1});

  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  ////////////////////////////////////
  //Ort::MemoryInfo output_mem_info{"Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault};
  /* Ort::MemoryInfo info_cuda("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault); */
  /* Ort::Session session(session_cache->ort_env, "/home/si/Desktop/artifacts/training_model.onnx", session_cache->session_options); */
  /* Ort::Allocator gpu_allocator(session, info_cuda); */
  /* auto ort_value = Ort::Value::CreateTensor(gpu_allocator, */
  /* 					    input_shape.data(), */
  /* 					    input_shape.size(), */
  /* 					    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT); */
  


  
  std::vector<Ort::Value> user_inputs; 

  user_inputs.emplace_back(Ort::Value::CreateTensor(memory_info,
						    batches,
						    1 * INSTANCE_WORDS_LENGTH * INSTANCE_WORDS_DIM * sizeof(float),
						    input_shape.data(),
						    input_shape.size(),
						    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

  /////////////////////
  // io_binding.BindInput("input", user_inputs);

  

  // Labels batched
  user_inputs.emplace_back(Ort::Value::CreateTensor(memory_info,
						    label_at,
						    1 * sizeof(int64_t),
						    labels_shape.data(),
						    labels_shape.size(),
						    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));
  //////////////////////////////
  // io_binding.BindOutput("output", output_mem_info);
  
  // Run the train step and execute the forward + loss + backward.
 float loss = *(session_cache->training_session.TrainStep(user_inputs)
		 .front()
		 .GetTensorMutableData<float>());

  session_cache->training_session.OptimizerStep();
  session_cache->training_session.LazyResetGrad();

  return loss;
};




#endif 
