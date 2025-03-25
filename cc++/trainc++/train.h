#ifndef TRAIN_H
#define TRAIN_H

#include "iostream"
#include "session.h"

float train_step(sessions::SessionCache *session_cache, float *batches, std::array<float, TARGET_CLASS_NUMBER> labels) {

  
  const std::vector<int64_t> input_shape({1, INSTANCE_WORDS_LENGTH, INSTANCE_WORDS_DIM});

  int64_t label_at[1] = {};
  label_at[0] = std::distance(labels.begin(), std::max_element(labels.begin(), labels.end()));
  const std::vector<int64_t> labels_shape({1});

  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  
  std::vector<Ort::Value> user_inputs; 
  user_inputs.emplace_back(Ort::Value::CreateTensor(memory_info,
						    batches,
						    1 * INSTANCE_WORDS_LENGTH * INSTANCE_WORDS_DIM * sizeof(float),
						    input_shape.data(),
						    input_shape.size(),
						    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));

  user_inputs.emplace_back(Ort::Value::CreateTensor(memory_info,
						    label_at,
						    1 * sizeof(int64_t),
						    labels_shape.data(),
						    labels_shape.size(),
						    ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));
  
 float loss = *(session_cache->training_session.TrainStep(user_inputs)
		.front()
		.GetTensorMutableData<float>());

  session_cache->training_session.OptimizerStep();
  session_cache->training_session.LazyResetGrad();

  return loss;
};


#endif 
