#include "iostream"
#include <fstream>
#include "session.h"
#include "train.h"
#include <chrono>
#include <unistd.h>
#include <string.h>
#include "infer.h"
#include <stdlib.h>  


int main(int argc, char *argv[]) {
  auto start = std::chrono::high_resolution_clock::now();
  printf("You are doing ondevice training with C++ !\n");

  if (argc != 3){
    std::cout << "Please speicfy your argument as following order: \n\
                  Percentage of trainset: 40 or 60\n\
                  retraining: resuming or ondevice \n" << std::endl;
    return 0;
  }

  int instance_Nr = 0;
  char data_name[100] = "";
  strcat(data_name, DATA_PATH);
  strcat(data_name, argv[1]);
  strcat(data_name, "_conversation_");
  
  char label_name[100] = "";
  strcat(label_name, LABEL_PATH);
  strcat(label_name, argv[1]);
  strcat(label_name, "_conversation_");

  if (strcmp(argv[1], "40") == 0){
    if (strcmp(argv[2], "ondevice") == 0) {
      instance_Nr = atoi(INSTANCE_NUMBER_40ON);
      strcat(data_name, ONDEVICE_DATA_NAME);
      strcat(label_name, ONDEVICE_LABEL_NAME); 
    }else if (strcmp(argv[2], "resuming") == 0) {
      instance_Nr = atoi(INSTANCE_NUMBER_40RE);
      strcat(data_name, RESUME_DATA_NAME);
      strcat(label_name, RESUME_LABEL_NAME); 
    }else {
      std::cout << "Retraing can only be done for ondevice or resuming!"<< std::endl;
      return 0;
    }
  }else if (strcmp(argv[1], "60") == 0){
    if (strcmp(argv[2], "ondevice") == 0) {
      instance_Nr = atoi(INSTANCE_NUMBER_60ON);
      strcat(data_name, ONDEVICE_DATA_NAME);
      strcat(label_name, ONDEVICE_LABEL_NAME); 
    }else if (strcmp(argv[2], "resuming") == 0) {
      instance_Nr = atoi(INSTANCE_NUMBER_60RE);
      strcat(data_name, RESUME_DATA_NAME);
      strcat(label_name, RESUME_LABEL_NAME); 
    }else {
      std::cout << "Retraing can only be done for ondevice or resuming!"<< std::endl;
      return 0;
    }
  }else {
    std::cout << "Percentage of trainset can only be specified by 40 or 60!"<< std::endl;
    return 0;
  }

  std::cout << "The instance number for retraining is "<< instance_Nr << std::endl;
  std::cout << "The instance number for inference is "<< INSTANCE_NUMBER_INFER << std::endl;

  if (access(data_name, F_OK) == 0) {
    printf("File exists: %s\n", data_name);
  } else {
    printf("File does not exist: %s\n", data_name);
    return 0;
  }
  if (access(label_name, F_OK) == 0) {
    printf("File exists: %s\n", label_name);
  } else {
    printf("File does not exist: %s\n", label_name);
    return 0;
  }

  char infer_data_name[100] = "";
  strcat(infer_data_name, DATA_PATH);
  strcat(infer_data_name, argv[1]);
  strcat(infer_data_name, "_conversation_");
  strcat(infer_data_name, INFER_DATA_NAME);
  if (access(infer_data_name, F_OK) == 0) {
    printf("File exists: %s\n", infer_data_name);
  } else {
    printf("File does not exist: %s\n", infer_data_name);
    return 0;
  }
  char infer_label_name[100] = "";
  strcat(infer_label_name, LABEL_PATH);
  strcat(infer_label_name, argv[1]);
  strcat(infer_label_name, "_conversation_");
  strcat(infer_label_name, INFER_LABEL_NAME); 
  if (access(infer_label_name, F_OK) == 0) {
    printf("File exists: %s\n", infer_label_name);
  } else {
    printf("File does not exist: %s\n", infer_label_name);
    return 0;
  }
  

  sessions::SessionCache session_cache = {
    CHECKPOINT_PATH,
    TRAINING_MODEL_PATH,
    EVAL_MODEL_PATH,
    OPTIMIZER_MODEL_PATH,
    CACHE_DIR_PATH
  };
  float model_input[INSTANCE_WORDS_LENGTH * INSTANCE_WORDS_DIM];  
  std::array<float, TARGET_CLASS_NUMBER> model_output{};


  float infer_data[INSTANCE_WORDS_LENGTH * INSTANCE_WORDS_DIM];  
  std::array<float, TARGET_CLASS_NUMBER> infer_label{};

  // set learning rate
  float learning_rate =  session_cache.training_session.GetLearningRate();
  printf("Old learning rate is : %f\n", learning_rate);
  
  float new_learning_rate = atof(LEARNING_RATE);
  // float new_learning_rate = atof(NEW_LEARNING_RATE);
  session_cache.training_session.SetLearningRate(new_learning_rate);
  new_learning_rate =  session_cache.training_session.GetLearningRate();
  printf("New learning rate is : %.7f\n", new_learning_rate);

  
  std::ifstream fin_label(label_name, std::ios::binary);
  std::ifstream fin_data(data_name, std::ios::binary);
  std::ifstream fin_infer_label(infer_label_name, std::ios::binary);
  std::ifstream fin_infer_data(infer_data_name, std::ios::binary);
  
  for (int epoch = 1; epoch<= EPOCHS; epoch++){
  float loss = 0;
  float loss_infer = 0;
    for (int i = 0; i<instance_Nr; i++) {
      fin_data.read(reinterpret_cast<char *>(&model_input), sizeof(float)*INSTANCE_WORDS_LENGTH*INSTANCE_WORDS_DIM);
      fin_label.read(reinterpret_cast<char *>(&model_output), sizeof(float)*TARGET_CLASS_NUMBER);
      loss += train_step(&session_cache, model_input, model_output);
    };
    fin_data.clear();
    fin_data.seekg(0, std::ios::beg);
    fin_label.clear();
    fin_label.seekg(0, std::ios::beg);
    std::cout<< "Loss for retraining at epoch "<< epoch << " is " << loss/instance_Nr << std::endl;
    session_cache.training_session.ExportModelForInferencing(session_cache.artifact_paths.inference_model_path.c_str(), {"output"});

    for (int i = 0; i < atoi(INSTANCE_NUMBER_INFER); i++) {
      fin_infer_data.read(reinterpret_cast<char *>(&infer_data), sizeof(float)*INSTANCE_WORDS_LENGTH*INSTANCE_WORDS_DIM);
      fin_infer_label.read(reinterpret_cast<char *>(&infer_label), sizeof(float) * TARGET_CLASS_NUMBER);
      loss_infer += infer(&session_cache, infer_data, infer_label);
    }
    fin_infer_data.clear();
    fin_infer_data.seekg(0, std::ios::beg);
    fin_infer_label.clear();
    fin_infer_label.seekg(0, std::ios::beg);
    std::cout<< "Loss for inference ---------------- is " << -1*loss_infer/atoi(INSTANCE_NUMBER_INFER)  << std::endl;
  };
  
  session_cache.training_session.ExportModelForInferencing(session_cache.artifact_paths.inference_model_path.c_str(), {"output"});

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout << "Time taken: " << duration/1000000 << " seconds" << std::endl;

  return 0;
}
