add_executable(cuda12trainc++ "cuda12trainc++.cc")
target_include_directories(cuda12trainc++ PRIVATE "${ONNXPATHTRAIN}/include")
target_link_libraries(cuda12trainc++ PRIVATE  "${ONNXPATHTRAIN}/lib/libonnxruntime.so")

target_compile_definitions(cuda12trainc++ PRIVATE CHECKPOINT_PATH=\"${CHECKPOINT_PATH}\")
target_compile_definitions(cuda12trainc++ PRIVATE TRAINING_MODEL_PATH=\"${TRAINING_MODEL_PATH}\")
target_compile_definitions(cuda12trainc++ PRIVATE EVAL_MODEL_PATH=\"${EVAL_MODEL_PATH}\")
target_compile_definitions(cuda12trainc++ PRIVATE OPTIMIZER_MODEL_PATH=\"${OPTIMIZER_MODEL_PATH}\")
target_compile_definitions(cuda12trainc++ PRIVATE INFERENCE_PATH=\"${INFERENCE_PATH}\")
target_compile_definitions(cuda12trainc++ PRIVATE CACHE_DIR_PATH=\"${CACHE_DIR_PATH}\")
target_compile_definitions(cuda12trainc++ PRIVATE EPOCHS=${EPOCHS})
target_compile_definitions(cuda12trainc++ PRIVATE INSTANCE_NUMBER=${INSTANCE_NUMBER})
target_compile_definitions(cuda12trainc++ PRIVATE INSTANCE_WORDS_LENGTH=${INSTANCE_WORDS_LENGTH})
target_compile_definitions(cuda12trainc++ PRIVATE INSTANCE_WORDS_DIM=${INSTANCE_WORDS_DIM})
target_compile_definitions(cuda12trainc++ PRIVATE TARGET_CLASS_NUMBER=${TARGET_CLASS_NUMBER})
target_compile_definitions(cuda12trainc++ PRIVATE DATA_PATH=\"${DATA_PATH}\")
target_compile_definitions(cuda12trainc++ PRIVATE LABEL_PATH=\"${LABEL_PATH}\")
