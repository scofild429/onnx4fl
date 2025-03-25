add_executable(trainc++ "trainc++.cc")
target_include_directories(trainc++ PRIVATE "${ONNXPATHTRAIN}/include")
target_link_libraries(trainc++ PRIVATE "${ONNXPATHTRAIN}/lib/libonnxruntime.so")

target_compile_definitions(trainc++ PRIVATE CHECKPOINT_PATH=\"${CHECKPOINT_PATH}\")
target_compile_definitions(trainc++ PRIVATE TRAINING_MODEL_PATH=\"${TRAINING_MODEL_PATH}\")
target_compile_definitions(trainc++ PRIVATE EVAL_MODEL_PATH=\"${EVAL_MODEL_PATH}\")
target_compile_definitions(trainc++ PRIVATE OPTIMIZER_MODEL_PATH=\"${OPTIMIZER_MODEL_PATH}\")
target_compile_definitions(trainc++ PRIVATE INFERENCE_PATH=\"${INFERENCE_PATH}\")
target_compile_definitions(trainc++ PRIVATE CACHE_DIR_PATH=\"${CACHE_DIR_PATH}\")
target_compile_definitions(trainc++ PRIVATE EPOCHS=${EPOCHS})
target_compile_definitions(trainc++ PRIVATE INSTANCE_NUMBER=${INSTANCE_NUMBER})
target_compile_definitions(trainc++ PRIVATE INSTANCE_WORDS_LENGTH=${INSTANCE_WORDS_LENGTH})
target_compile_definitions(trainc++ PRIVATE INSTANCE_WORDS_DIM=${INSTANCE_WORDS_DIM})
target_compile_definitions(trainc++ PRIVATE TARGET_CLASS_NUMBER=${TARGET_CLASS_NUMBER})
target_compile_definitions(trainc++ PRIVATE DATA_PATH=\"${DATA_PATH}\")
target_compile_definitions(trainc++ PRIVATE LABEL_PATH=\"${LABEL_PATH}\")
target_compile_definitions(trainc++ PRIVATE RESUME_DATA_NAME=\"${RESUME_DATA_NAME}\")
target_compile_definitions(trainc++ PRIVATE RESUME_LABEL_NAME=\"${RESUME_LABEL_NAME}\")
target_compile_definitions(trainc++ PRIVATE ONDEVICE_DATA_NAME=\"${ONDEVICE_DATA_NAME}\")
target_compile_definitions(trainc++ PRIVATE ONDEVICE_LABEL_NAME=\"${ONDEVICE_LABEL_NAME}\")

target_compile_definitions(trainc++ PRIVATE LEARNING_RATE=\"${LEARNING_RATE}\")
target_compile_definitions(trainc++ PRIVATE NEW_LEARNING_RATE=\"${NEW_LEARNING_RATE}\")

target_compile_definitions(trainc++ PRIVATE ONDEVICE_INFERENCE_INPUT=\"${ONDEVICE_INFERENCE_INPUT}\")
target_compile_definitions(trainc++ PRIVATE ONDEVICE_INFERENCE_OUTPUT=\"${ONDEVICE_INFERENCE_OUTPUT}\")
target_compile_definitions(trainc++ PRIVATE INFER_LABEL_NAME=\"${INFER_LABEL_NAME}\")
target_compile_definitions(trainc++ PRIVATE INFER_DATA_NAME=\"${INFER_DATA_NAME}\")
target_compile_definitions(trainc++ PRIVATE INSTANCE_NUMBER_INFER=\"${INSTANCE_NUMBER_INFER}\")
target_compile_definitions(trainc++ PRIVATE INSTANCE_NUMBER_60ON=\"${INSTANCE_NUMBER_60ON}\")
target_compile_definitions(trainc++ PRIVATE INSTANCE_NUMBER_60RE=\"${INSTANCE_NUMBER_60RE}\")
target_compile_definitions(trainc++ PRIVATE INSTANCE_NUMBER_40ON=\"${INSTANCE_NUMBER_40ON}\")
target_compile_definitions(trainc++ PRIVATE INSTANCE_NUMBER_40RE=\"${INSTANCE_NUMBER_40RE}\")
# target_compile_definitions(trainc++ PRIVATE =\"${}\")






