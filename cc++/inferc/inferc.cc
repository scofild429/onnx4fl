#include <algorithm>
#include <array>
#include <assert.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>
#include <time.h>
#include "./utils.h"
#include "onnxruntime_c_api.h"
#include <unistd.h>

const OrtApi *g_ort = NULL;

int run_inference(OrtSession *session, const char *input_names[], const char *output_names[], char *data_name, char *label_name) {
  OrtMemoryInfo *memory_info;
  ORT_ABORT_ON_ERROR(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

  // model output
  std::array<float, TARGET_CLASS_NUMBER> model_output{};
  const int64_t model_output_shape[] = {1, TARGET_CLASS_NUMBER};
  const size_t model_output_shape_len = sizeof(model_output_shape) / sizeof(model_output_shape[0]);
  OrtValue *output_tensor = NULL;
  ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info,
							   model_output.data(),
							   model_output.size() * sizeof(float),
							   model_output_shape,
							   model_output_shape_len,
							   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
							   &output_tensor));

  // model input
  float model_input[INSTANCE_WORDS_LENGTH * INSTANCE_WORDS_DIM];
  size_t model_input_len = 1 * INSTANCE_WORDS_LENGTH * INSTANCE_WORDS_DIM * sizeof(float);
  int64_t model_input_shape[] = {1, INSTANCE_WORDS_LENGTH, INSTANCE_WORDS_DIM};
  size_t model_input_shape_len = sizeof(model_input_shape) / sizeof(model_input_shape[0]);
  OrtValue *input_tensor = NULL;
  
  int data_index{0};
  std::array<float, INSTANCE_NUMBER> predict{};
  std::ifstream fin_data(data_name, std::ios::binary);
  for (int i = 0; i < INSTANCE_NUMBER; i++) {
    fin_data.read(reinterpret_cast<char *>(&model_input), sizeof(float)*INSTANCE_WORDS_LENGTH*INSTANCE_WORDS_DIM);
    ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info,
							     model_input,
							     model_input_len,
							     model_input_shape,
							     model_input_shape_len,
							     ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
							     &input_tensor));
    // model run
    ORT_ABORT_ON_ERROR(g_ort->Run(session,
				  NULL,
				  input_names,
                                  (const OrtValue *const *)&input_tensor,
				  1,
				  output_names,
				  1,
				  &output_tensor)
		       );
    // result
    softmax(model_output);
    data_index = std::distance(model_output.begin(), std::max_element(model_output.begin(), model_output.end()));
    predict[i] = data_index;
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
    //    std::cout << labels[i] << ":" << predict[i] << std::endl;
    if (labels[i] == predict[i])
      corrected++;
  }

  std::cout << "The corrected predicted instances are " << corrected
            << " from totally " << INSTANCE_NUMBER
	    << " input instances, the accuracy is "
	    << float(corrected) / 1556 << std::endl;

  g_ort->ReleaseMemoryInfo(memory_info);
  g_ort->ReleaseValue(output_tensor);
  g_ort->ReleaseValue(input_tensor);
  
  return 0;
}

int main(int argc, char *argv[]) {
  time_t start, end;
  time(&start);
  
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
    printf("You are using Dynamo ONNX for inference in C!\n");
    model_path =(ORTCHAR_T *)ONNX_DYNAMO_INFERENCE_PATH;
    input_names[0] = ONNX_DYNAMO_INFERENCE_INPUT;
    output_names[0] = ONNX_DYNAMO_INFERENCE_OUTPUT;
  }else if(strcmp(argv[1], "2") == 0) {
    printf("You are using Torchscript ONNX for inference in C!\n");
    model_path = (ORTCHAR_T *)ONNX_TORCHSCRIPT_INFERENCE_PATH;
    input_names[0] = ONNX_TORCHSCRIPT_INFERENCE_INPUT;
    output_names[0] = ONNX_TORCHSCRIPT_INFERENCE_OUTPUT;
  }else if(strcmp(argv[1], "3") == 0) {
    printf("You are doing inference with 20 epochs resuming trained ONNX for conversation inference with C!\n");
    model_path = (ORTCHAR_T *)ONNX_C20RESUMING_INFERENCE_PATH;
    input_names[0] = ONDEVICE_INFERENCE_INPUT;
    output_names[0] = ONDEVICE_INFERENCE_OUTPUT;
  }else if(strcmp(argv[1], "4") == 0) {
    printf("You are doing inference with 40 epochs resuming trained ONNX for conversation inference with C!\n");
    model_path = (ORTCHAR_T *)ONNX_C40RESUMING_INFERENCE_PATH;
    input_names[0] = ONDEVICE_INFERENCE_INPUT;
    output_names[0] = ONDEVICE_INFERENCE_OUTPUT;
  }else if(strcmp(argv[1], "5") == 0) {
    printf("You are doing inference with 20 epochs ondevice trained ONNX for conversation inference with C!\n");
    model_path = (ORTCHAR_T *)ONNX_C20ONDEVICE_INFERENCE_PATH;
    input_names[0] = ONDEVICE_INFERENCE_INPUT;
    output_names[0] = ONDEVICE_INFERENCE_OUTPUT;
  }else if(strcmp(argv[1], "6") == 0) {
    printf("You are doing inference with 40 epochs ondevice trained ONNX for conversation inference with C!\n");
    model_path = (ORTCHAR_T *)ONNX_C40ONDEVICE_INFERENCE_PATH;
    input_names[0] = ONDEVICE_INFERENCE_INPUT;
    output_names[0] = ONDEVICE_INFERENCE_OUTPUT;
  }else if(strcmp(argv[1], "7") == 0) {
    printf("You are doing inference with 20 epochs resuming trained ONNX for question inference with C!\n");
    model_path = (ORTCHAR_T *)ONNX_Q20RESUMING_INFERENCE_PATH;
    input_names[0] = ONDEVICE_INFERENCE_INPUT;
    output_names[0] = ONDEVICE_INFERENCE_OUTPUT;
  }else if(strcmp(argv[1], "8") == 0) {
    printf("You are doing inference with 40 epochs resuming trained ONNX for question inference with C!\n");
    model_path = (ORTCHAR_T *)ONNX_Q40RESUMING_INFERENCE_PATH;
    input_names[0] = ONDEVICE_INFERENCE_INPUT;
    output_names[0] = ONDEVICE_INFERENCE_OUTPUT;
  }else if(strcmp(argv[1], "9") == 0) {
    printf("You are doing inference with 20 epochs ondevice trained ONNX for question inference with C!\n");
    model_path = (ORTCHAR_T *)ONNX_Q20ONDEVICE_INFERENCE_PATH;
    input_names[0] = ONDEVICE_INFERENCE_INPUT;
    output_names[0] = ONDEVICE_INFERENCE_OUTPUT;
  }else if(strcmp(argv[1], "10") == 0) {
    printf("You are doing inference with 40 epochs ondevice trained ONNX for question inference with C!\n");
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

  // Get ORT version
  g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  
  // get ortEnv
  OrtEnv *env;
  ORT_ABORT_ON_ERROR(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));

  // get OrtSessionOption
  OrtSessionOptions *session_options;
  ORT_ABORT_ON_ERROR(g_ort->CreateSessionOptions(&session_options));

  // get OrtSession
  OrtSession *session;
  ORT_ABORT_ON_ERROR(g_ort->CreateSession(env, model_path, session_options, &session));

  
  //run inference
  int ret = run_inference(session, input_names, output_names, data_name, label_name);

  // release
  g_ort->ReleaseSessionOptions(session_options);
  g_ort->ReleaseSession(session);
  g_ort->ReleaseEnv(env);

  time(&end);
  double time_diff = difftime(end, start);
  printf("Time consumption is %.2f seconds.\n", time_diff);
  if (ret == 0) {
    std::cout << "Done at: " << PROJECT_BUILD_DATE << std::endl;
  }
  
  return 0;
}



