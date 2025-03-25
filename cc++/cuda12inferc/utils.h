
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


template <typename T> static void softmax(T &input) {
  float rowmax = *std::max_element(input.begin(), input.end());
  std::vector<float> y(input.size());
  float sum = 0.0f;
  for (size_t i = 0; i != input.size(); ++i) {
    sum += y[i] = std::exp(input[i] - rowmax);
  }
  for (size_t i = 0; i != input.size(); ++i) {
    input[i] = y[i] / sum;
  }
}


#endif
