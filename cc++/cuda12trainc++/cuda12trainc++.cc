#include "iostream"
#include <fstream>
#include <chrono>
#include "session.h"
#include "train.h"


int main() {
  auto start = std::chrono::high_resolution_clock::now();
  printf("You are doing ondevice training in C++ with CUDA, but cuda code does not work!\n");
  
  sessions::SessionCache session_cache = {
    CHECKPOINT_PATH,
    TRAINING_MODEL_PATH,
    EVAL_MODEL_PATH,
    OPTIMIZER_MODEL_PATH,
    CACHE_DIR_PATH
  };
  float model_input[INSTANCE_WORDS_LENGTH * INSTANCE_WORDS_DIM];  
  std::array<float, TARGET_CLASS_NUMBER> model_output{};

  // set learning rate
  float learning_rate =  session_cache.training_session.GetLearningRate();
  printf("Old learning rate is : %f\n", learning_rate);
  
  float new_learning_rate = 0.0000005;
  session_cache.training_session.SetLearningRate(new_learning_rate);
  new_learning_rate =  session_cache.training_session.GetLearningRate();
  printf("New learning rate is : %.7f\n", new_learning_rate);

  
  std::ifstream fin_label(LABEL_PATH, std::ios::binary);
  std::ifstream fin_data(DATA_PATH, std::ios::binary);
  float loss;
  for (int epoch = 1; epoch<= EPOCHS; epoch++){
    for (int i = 0; i<INSTANCE_NUMBER ; i++) {
      fin_data.read(reinterpret_cast<char *>(&model_input), sizeof(float)*INSTANCE_WORDS_LENGTH*INSTANCE_WORDS_DIM);
      fin_label.read(reinterpret_cast<char *>(&model_output), sizeof(float)*TARGET_CLASS_NUMBER);

      loss += train_step(&session_cache, model_input, model_output);
    };
    std::cout<< "Loss at epoch "<< epoch << " is " << loss/INSTANCE_NUMBER << std::endl;
  };
  
  session_cache.training_session.ExportModelForInferencing(session_cache.artifact_paths.inference_model_path.c_str(), {"output"});


  
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout << "Time taken: " << duration/1000000 << " seconds" << std::endl;

  return 0;
}
