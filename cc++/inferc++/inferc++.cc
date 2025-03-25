#include <iostream>
#include <fstream>
#include "./utils.h"
#include <chrono>
#include <unistd.h>
#include <string.h>
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_c_api.h"


int main(int argc, char *argv[]) {
  auto start = std::chrono::high_resolution_clock::now();
  
  if (argc != 4){
    std::cout << "Please speicfy your argument as following order: \n \
                   ONNX type: 1 for Dynamo, 2 for Torchscript, 3 for ondevice trained\n \
                   Percentage of trainset: 40 or 60\n \
                   Evaluation: question or conversation.\n" << std::endl;
    return 0;
  }
  if (!((strcmp(argv[2], "40") == 0) || (strcmp(argv[2], "60") == 0))){
    std::cout << "Percentage of trainset can only be specified by 40 or 60!"<< std::endl;
    return 0;
  }
  if (!((strcmp(argv[3], "question") == 0) || (strcmp(argv[3], "conversation") == 0))){
    std::cout << "Evaluation can only be specified by question or conversation!"<< std::endl;
    return 0;
  }


  ORTCHAR_T *model_path;
  const char *input_names[1];
  const char *output_names[1];
  if (strcmp(argv[1], "1") == 0){
    printf("You are using Dynamo ONNX for inference in C++!\n");
    model_path =(ORTCHAR_T *)ONNX_DYNAMO_INFERENCE_PATH;
    input_names[0] = ONNX_DYNAMO_INFERENCE_INPUT;
    output_names[0] = ONNX_DYNAMO_INFERENCE_OUTPUT;
  }else if(strcmp(argv[1], "2") == 0) {
    printf("You are using Torchscript ONNX for inference in C++!\n");
    model_path = (ORTCHAR_T *)ONNX_TORCHSCRIPT_INFERENCE_PATH;
    input_names[0] = ONNX_TORCHSCRIPT_INFERENCE_INPUT;
    output_names[0] = ONNX_TORCHSCRIPT_INFERENCE_OUTPUT;
  }else if(strcmp(argv[1], "3") == 0) {
    printf("You are doing inference with 20 epochs resuming trained ONNX for conversation inference with C++!\n");
    model_path = (ORTCHAR_T *)ONNX_C20RESUMING_INFERENCE_PATH;
    input_names[0] = ONDEVICE_INFERENCE_INPUT;
    output_names[0] = ONDEVICE_INFERENCE_OUTPUT;
  }else if(strcmp(argv[1], "4") == 0) {
    printf("You are doing inference with 40 epochs resuming trained ONNX for conversation inference with C++!\n");
    model_path = (ORTCHAR_T *)ONNX_C40RESUMING_INFERENCE_PATH;
    input_names[0] = ONDEVICE_INFERENCE_INPUT;
    output_names[0] = ONDEVICE_INFERENCE_OUTPUT;
  }else if(strcmp(argv[1], "5") == 0) {
    printf("You are doing inference with 20 epochs ondevice trained ONNX for conversation inference with C++!\n");
    model_path = (ORTCHAR_T *)ONNX_C20ONDEVICE_INFERENCE_PATH;
    input_names[0] = ONDEVICE_INFERENCE_INPUT;
    output_names[0] = ONDEVICE_INFERENCE_OUTPUT;
  }else if(strcmp(argv[1], "6") == 0) {
    printf("You are doing inference with 40 epochs ondevice trained ONNX for conversation inference with C++!\n");
    model_path = (ORTCHAR_T *)ONNX_C40ONDEVICE_INFERENCE_PATH;
    input_names[0] = ONDEVICE_INFERENCE_INPUT;
    output_names[0] = ONDEVICE_INFERENCE_OUTPUT;
  }else if(strcmp(argv[1], "7") == 0) {
    printf("You are doing inference with 20 epochs resuming trained ONNX for question inference with C++!\n");
    model_path = (ORTCHAR_T *)ONNX_Q20RESUMING_INFERENCE_PATH;
    input_names[0] = ONDEVICE_INFERENCE_INPUT;
    output_names[0] = ONDEVICE_INFERENCE_OUTPUT;
  }else if(strcmp(argv[1], "8") == 0) {
    printf("You are doing inference with 40 epochs resuming trained ONNX for question inference with C++!\n");
    model_path = (ORTCHAR_T *)ONNX_Q40RESUMING_INFERENCE_PATH;
    input_names[0] = ONDEVICE_INFERENCE_INPUT;
    output_names[0] = ONDEVICE_INFERENCE_OUTPUT;
  }else if(strcmp(argv[1], "9") == 0) {
    printf("You are doing inference with 20 epochs ondevice trained ONNX for question inference with C++!\n");
    model_path = (ORTCHAR_T *)ONNX_Q20ONDEVICE_INFERENCE_PATH;
    input_names[0] = ONDEVICE_INFERENCE_INPUT;
    output_names[0] = ONDEVICE_INFERENCE_OUTPUT;
  }else if(strcmp(argv[1], "10") == 0) {
    printf("You are doing inference with 40 epochs ondevice trained ONNX for question inference with C++!\n");
    model_path = (ORTCHAR_T *)ONNX_Q40ONDEVICE_INFERENCE_PATH;
    input_names[0] = ONDEVICE_INFERENCE_INPUT;
    output_names[0] = ONDEVICE_INFERENCE_OUTPUT;
  }else {
      std::cout << "ONNX type can only between 1 to 10!"<< std::endl;
      return 0;
 }	

  if (access(model_path, F_OK) == 0) {
    printf("File exists: %s\n", model_path);
  } else {
    printf("File does not exist: %s.\n", model_path);
    return 0;
  }


  char data_name[100] = "";
  strcat(data_name, DATA_PATH);
  strcat(data_name, argv[2]);
  strcat(data_name, "_");
  strcat(data_name, argv[3]);
  strcat(data_name, "_");
  strcat(data_name, DATA_NAME);

  if (access(data_name, F_OK) == 0) {
    printf("File exists: %s\n", data_name);
  } else {
    printf("File does not exist: %s.\n", data_name);
    return 0;
  }

  char label_name[100] = "";
  strcat(label_name, LABEL_PATH);
  strcat(label_name, argv[2]);
  strcat(label_name, "_");
  strcat(label_name, argv[3]);
  strcat(label_name, "_");
  strcat(label_name, LABEL_NAME); 

  if (access(label_name, F_OK) == 0) {
    printf("File exists: %s\n", label_name);
  } else {
    printf("File does not exist: %s\n", label_name);
    return 0;
  }

  
  Ort::SessionOptions session_option = Ort::SessionOptions{};
  Ort::Env ort_env(ORT_LOGGING_LEVEL_WARNING, "ort personalize");
  Ort::Session ort_session{ort_env, model_path, session_option};
  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  
  float model_input[INSTANCE_WORDS_LENGTH * INSTANCE_WORDS_DIM];
  size_t model_input_len = 1 * INSTANCE_WORDS_LENGTH * INSTANCE_WORDS_DIM * sizeof(float);
  int64_t model_input_shape[] = {1, INSTANCE_WORDS_LENGTH, INSTANCE_WORDS_DIM};
  size_t model_input_shape_len = sizeof(model_input_shape) / sizeof(model_input_shape[0]);

  std::array<float, INSTANCE_NUMBER> predict{};
  std::vector<Ort::Value> input_values;
  std::ifstream fin_data(data_name, std::ios::binary);
  for (int i = 0; i < INSTANCE_NUMBER; i++) {
    fin_data.read(reinterpret_cast<char *>(&model_input), sizeof(float)*INSTANCE_WORDS_LENGTH*INSTANCE_WORDS_DIM);
    input_values.emplace_back(Ort::Value::CreateTensor(memory_info,
						       model_input,
						       model_input_len,
						       model_input_shape,
						       model_input_shape_len,
						       ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
    std::vector<Ort::Value> output_values;
    output_values.emplace_back(nullptr);

    ort_session.Run(Ort::RunOptions(),
		    input_names,
		    input_values.data(),
		    1,
		    output_names,
		    output_values.data(),
		    1);

    float *output = output_values.front().GetTensorMutableData<float>();
    std::vector<float> probabilities = Softmax(output, TARGET_CLASS_NUMBER);
    size_t best_index = std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()));
    predict[i] = best_index;
  }

  // get labels
  int label_at{0};
  std::array<float, TARGET_CLASS_NUMBER> label_index{};
  std::array<float, INSTANCE_NUMBER> labels{};
  std::ifstream fin_label(label_name, std::ios::binary);
  for (int i = 0; i < INSTANCE_NUMBER; i++) {
    fin_label.read(reinterpret_cast<char *>(&label_index), sizeof(float) * TARGET_CLASS_NUMBER);
    label_at = std::distance(label_index.begin(), std::max_element(label_index.begin(), label_index.end()));
    labels[i] = label_at;
  }
  // get corrected
  int corrected = 0;
  for (int i = 0; i < INSTANCE_NUMBER; i++) {
    if (labels[i] == predict[i])
      corrected++;
  }

  std::cout << "The corrected predicted instances are " << corrected
            << " from totally " << INSTANCE_NUMBER
	    << " input instances, the accuracy is "
	    << float(corrected) / INSTANCE_NUMBER << std::endl;

  
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout << "Time taken: " << duration/1000000 << " seconds" << std::endl;

  std::cout << "Done at: " << PROJECT_BUILD_DATE << std::endl;
  return 0;
}
