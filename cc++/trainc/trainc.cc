#include <string.h>
#include <stdio.h>
#include "onnxruntime_training_c_api.h"
#include "onnxruntime_c_api.h"
#include <iostream>
#include <fstream>
#include <time.h>
#include "utils.h"
#include <unistd.h>


void training(char *data_name, char *label_name) {
  const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  const OrtTrainingApi* g_ort_training = g_ort->GetTrainingApi(ORT_API_VERSION);

  OrtStatusPtr single;
  OrtEnv* env = NULL;
  single = g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env);
      
  OrtSessionOptions* session_options = NULL;
  single = g_ort->CreateSessionOptions(&session_options);
 
  OrtCheckpointState* state = NULL;
  char* checkpointPath = (char*) CHECKPOINT_PATH;
  single = g_ort_training->LoadCheckpoint(checkpointPath, &state);
  
  OrtTrainingSession* training_session = NULL;
  single =  g_ort_training->CreateTrainingSession(env,
						  session_options,
						  state,
						  TRAINING_MODEL_PATH,
						  EVAL_MODEL_PATH,
						  OPTIMIZER_MODEL_PATH,
						  &training_session);


  
  float learning_rate;
  single =  g_ort_training->GetLearningRate(training_session, &learning_rate );
  printf("Old learning rate is : %f\n", learning_rate);
  float new_learning_rate = atof(LEARNING_RATE);
  // float new_learning_rate = atof(NEW_LEARNING_RATE);
  
  single =  g_ort_training->SetLearningRate(training_session, new_learning_rate);
  single =  g_ort_training->GetLearningRate(training_session, &learning_rate );
  printf("New learning rate is : %.7f\n", learning_rate);
  
  single = g_ort_training->SetSeed(40);
  
  OrtMemoryInfo *memory_info;
  single = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);

  OrtRunOptions* run_option;

  float model_input[INSTANCE_WORDS_LENGTH * INSTANCE_WORDS_DIM];
  size_t model_input_len = 1 * INSTANCE_WORDS_LENGTH * INSTANCE_WORDS_DIM * sizeof(float);
  int64_t model_input_shape[] = {1, INSTANCE_WORDS_LENGTH, INSTANCE_WORDS_DIM};
  size_t model_input_shape_len = sizeof(model_input_shape) / sizeof(model_input_shape[0]);
  OrtValue *input_tensor = NULL;

  std::array<float, TARGET_CLASS_NUMBER> model_output{};
  OrtValue *output_tensor = NULL;

  
  std::ifstream fin_data(data_name, std::ios::binary);
  std::ifstream fin_label(label_name, std::ios::binary);

  for (int epoch = 0; epoch<EPOCHS; epoch++) {
    for (int i = 0; i<INSTANCE_NUMBER; i++) {
      fin_data.read(reinterpret_cast<char *>(&model_input), sizeof(float)*INSTANCE_WORDS_LENGTH*INSTANCE_WORDS_DIM);
      single = g_ort->CreateTensorWithDataAsOrtValue(memory_info,
						     model_input,
						     model_input_len,
						     model_input_shape,
						     model_input_shape_len,
						     ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
						     &input_tensor);

      fin_label.read(reinterpret_cast<char *>(&model_output), sizeof(float)*TARGET_CLASS_NUMBER);
      int label_at[1];
      int64_t label_shape[] = {1};
      label_at[0] = std::distance(model_output.begin(), std::max_element(model_output.begin(), model_output.end()));
      //      std::cout << "Lable is: "<<  label_at[0] << std::endl;
      ORT_ABORT_ON_ERROR(g_ort->CreateTensorWithDataAsOrtValue(memory_info,
							       label_at,
							       sizeof(int),
							       label_shape,
							       1,
							       ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
							       &output_tensor));
      g_ort_training->TrainStep(training_session,
				run_option,
				1,
				(const OrtValue *const *) input_tensor,
				1,
				&output_tensor);

      single = g_ort_training->OptimizerStep(training_session, run_option);
      single = g_ort_training->LazyResetGrad(training_session);
    };
  };
  single = g_ort_training->SaveCheckpoint(state, CHECKPOINT_PATH, false);

  // std::string name = "output";
  // ORT_ABORT_ON_ERROR(g_ort_training-> ExportModelForInferencing(training_session,
  // 								(const char *) inference_path,
  // 								1,
  // 								(const char *const *) name.c_str()
  // 								));
  
  //  std::cout<< "After export "<< &single << std::endl;
  g_ort_training->ReleaseTrainingSession(training_session);
  g_ort_training->ReleaseCheckpointState(state);
  g_ort->ReleaseSessionOptions(session_options);
  g_ort->ReleaseEnv(env);
  g_ort->ReleaseMemoryInfo(memory_info);
  //g_ort->ReleaseRunOptions(run_option);
  g_ort->ReleaseValue(input_tensor);
  g_ort->ReleaseValue(output_tensor);
};

int main(int argc, char *argv[]) {
  time_t start, end;
  time(&start);
  printf("You are doing ondevice training with C, but this can not save ONNX for inference!\n");


  if (argc != 3){
    std::cout << "Please speicfy your argument as following order: \n\
                  Percentage of trainset: 40 or 60\n\
                  Training with conversation: conversation.\n" << std::endl;
    return 0;
  }
  if (!((strcmp(argv[1], "40") == 0) || (strcmp(argv[1], "60") == 0))){
    std::cout << "Percentage of trainset can only be specified by 40 or 60!"<< std::endl;
    return 0;
  }
  if (strcmp(argv[2], "conversation") != 0){
    std::cout << "Retraing should use conversation!"<< std::endl;
    return 0;
  }

  char data_name[100] = "";
  strcat(data_name, DATA_PATH);
  strcat(data_name, argv[1]);
  strcat(data_name, "_");
  strcat(data_name, argv[2]);
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
  strcat(label_name, argv[1]);
  strcat(label_name, "_");
  strcat(label_name, argv[2]);
  strcat(label_name, "_");
  strcat(label_name, LABEL_NAME); 
  if (access(label_name, F_OK) == 0) {
    printf("File exists: %s\n", label_name);
  } else {
    printf("File does not exist: %s\n", label_name);
    return 0;
  }


  training(data_name, label_name);
  
  time(&end);
  double time_diff = difftime(end, start);
  printf("Time consumption is %.2f seconds.\n", time_diff);
  return 0;
}
