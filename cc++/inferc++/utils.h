
#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <vector>
#include <iostream>
#include <cmath>
#include <array>


#define ORT_ABORT_ON_ERROR(expr)                                               \
  do {                                                                         \
    OrtStatus *onnx_status = (expr);                                           \
    if (onnx_status != NULL) {                                                 \
      const char *msg = g_ort->GetErrorMessage(onnx_status);                   \
      std::cout << msg << std::endl;                                           \
      g_ort->ReleaseStatus(onnx_status);                                       \
      abort();                                                                 \
    }                                                                          \
  } while (0);


std::vector<float> Softmax(float *logits, size_t num_logits) {
  std::vector<float> probabilities(num_logits, 0);
  float sum = 0;
  for (size_t i = 0; i < num_logits; ++i) {
    probabilities[i] = exp(logits[i]);
    sum += probabilities[i];
  }

  if (sum != 0.0f) {
    for (size_t i = 0; i < num_logits; ++i) {
      probabilities[i] /= sum;
    }
  }

  return probabilities;
}


#endif
