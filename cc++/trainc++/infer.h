#ifndef INFER_H
#define INFER_H

#include <iostream>
#include <fstream>
#include "./utils.h"
#include <chrono>
#include <unistd.h>
#include <string.h>
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_c_api.h"
#include "session.h"
#include <array>
#include <cmath>


float infer(sessions::SessionCache* session_cache, float* infer_data, std::array<float, TARGET_CLASS_NUMBER>& infer_label) {
  const char *input_names[1] = {ONDEVICE_INFERENCE_INPUT};
  const char *output_names[1] = {ONDEVICE_INFERENCE_OUTPUT};
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  

  Ort::Session inference{session_cache->ort_env, INFERENCE_PATH, session_cache->session_options};
  
  //  float model_input[INSTANCE_WORDS_LENGTH * INSTANCE_WORDS_DIM];
  size_t model_input_len = 1 * INSTANCE_WORDS_LENGTH * INSTANCE_WORDS_DIM * sizeof(float);
  int64_t model_input_shape[] = {1, INSTANCE_WORDS_LENGTH, INSTANCE_WORDS_DIM};
  size_t model_input_shape_len = sizeof(model_input_shape) / sizeof(model_input_shape[0]);

  std::vector<Ort::Value> input_values;

  input_values.emplace_back(Ort::Value::CreateTensor(memory_info,
						     infer_data,
						     model_input_len,
						     model_input_shape,
						     model_input_shape_len,
						     ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
  std::vector<Ort::Value> output_values;
  output_values.emplace_back(nullptr);

  inference.Run(Ort::RunOptions(),
		input_names,
		input_values.data(),
		1,
		output_names,
		output_values.data(),
		1);

  float *output = output_values.front().GetTensorMutableData<float>();
  std::vector<float> probabilities = Softmax(output, TARGET_CLASS_NUMBER);

  float loss = 0;
  for (int i = 0; i < TARGET_CLASS_NUMBER; i++) {
    loss += infer_label[i]*log(probabilities[i]);
  }
  
  return loss;
}


#endif


