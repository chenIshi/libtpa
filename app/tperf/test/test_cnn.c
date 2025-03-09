#include <stdio.h>
#include "../onnx/include/onnxruntime_c_api.h"

int offrac_cnn(uint32_t *buf, int size, int offrac_size, int offrac_args) {
    OrtEnv* env;
    OrtStatus* status = OrtCreateEnv(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntimeModel", &env);

    if (status != NULL) {
        fprintf(stderr, "Error creating ONNX environment: %s\n", OrtGetErrorMessage(status));
        return -1;
    }

    // Session options and model loading
    OrtSessionOptions* session_options;
    OrtCreateSessionOptions(&session_options);

    OrtSession* session;
    status = OrtCreateSession(env, "./model/small_CNN.onnx", session_options, &session);
    if (status != NULL) {
        fprintf(stderr, "Error loading model: %s\n", OrtGetErrorMessage(status));
        return -1;
    }

    // Prepare input data (3D u_int8 array, size 3x64x64)
    u_int8_t input[3*64*64] = {0};
    int64_t input_dims[] = {1, 64, 64}; // The input dimensions: batch size, height, width
    OrtMemoryInfo* memory_info;
    OrtCreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);

    // Create input tensor
    OrtValue* input_tensor;
    status = OrtCreateTensorWithDataAsOrtValue(memory_info, input_data, sizeof(input_data), input_dims, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
    if (status != NULL) {
        const char* error_message = OrtGetErrorMessage(status);
        printf("Error creating input tensor: %s\n", error_message);
        return 1;
    }

    // Run inference
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    OrtValue* output_tensor;
    status = OrtRun(session, NULL, input_names, &input_tensor, 1, output_names, 1, &output_tensor);
    if (status != NULL) {
        fprintf(stderr, "Error running inference: %s\n", OrtGetErrorMessage(status));
        return -1;
    }

    // Get and print the result
    float* output_data;
    status = OrtGetTensorMutableData(output_tensor, (void**)&output_data);
    if (status != NULL) {
        fprintf(stderr, "Error getting output tensor data: %s\n", OrtGetErrorMessage(status));
        return -1;
    }

    printf("Inference result: %f\n", output_data[0]);

    // Cleanup
    OrtReleaseValue(input_tensor);
    OrtReleaseValue(output_tensor);
    OrtReleaseSession(session);
    OrtReleaseSessionOptions(session_options);
    OrtReleaseEnv(env);

    return 0;
}