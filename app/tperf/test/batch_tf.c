#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <float.h>
#include <inttypes.h>

// #include "../libtensorflow/include/tensorflow/c/c_api.h"
#include "tensorflow/c/c_api.h"

#define IMAGE_SIZE (64 * 64 * 3)
#define NUM_IMAGES 10

void check_status(TF_Status* status) {
    if (TF_GetCode(status) != TF_OK) {
        printf("Error: %s\n", TF_Message(status));
        exit(1);
    }
}

void NoOpDeallocator(void* data, size_t a, void* b) {}

int offrac_cnn(uint32_t *buf, int size, int offrac_size, int offrac_args) {
    // Load the TensorFlow model (SavedModel format)
    TF_Graph* graph = TF_NewGraph();
    TF_Status* status = TF_NewStatus();

    // Define session options and load the SavedModel
    TF_SessionOptions* session_opts = TF_NewSessionOptions();
    TF_Buffer* run_options = NULL;
    const char* tags = "serve";
    TF_Session* session = TF_LoadSessionFromSavedModel(session_opts, run_options, "saved_model", &tags, 1, graph, NULL, status);
    check_status(status);

    // Prepare input tensor (64x64 RGB image)
    if (size % (IMAGE_SIZE * sizeof(float)) != 0) {
        fprintf(stderr, "Invalid buffer size for CNN\n");
        return -1;
    }

    int num_images = size / (IMAGE_SIZE * sizeof(float));
    float* input_data = (float *)malloc(size);
    if (input_data == NULL) {
        fprintf(stderr, "Failed to allocate input data\n");
        return -1;
    }
    printf("hello 46\n");
    // Copy data from buf to input_data
    memcpy(input_data, buf, size);
    int64_t input_dims[] = {num_images, 64, 64, 3};  // Shape of the input image: (num_images, 64, 64, 3)

    // Create the input tensor
    TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT, input_dims, 4, input_data, size, &NoOpDeallocator, NULL);
    printf("hello 53\n");
    // Prepare output tensor
    //****** Get input tensor
    int NumInputs = 1;
    TF_Output* Input = (TF_Output*)malloc(sizeof(TF_Output) * NumInputs);

    TF_Output t0 = {TF_GraphOperationByName(graph, "serving_default_input_1"), 0};

    //********* Get Output tensor
    int NumOutputs = 1;
    TF_Output* Output = (TF_Output*)malloc(sizeof(TF_Output) * NumOutputs);

    TF_Output t2 = {TF_GraphOperationByName(graph, "StatefulPartitionedCall"), 0};

    if (t0.oper == NULL || t2.oper == NULL) {
        fprintf(stderr, "Failed to get input/output tensor\n");
        return -1;
    }
    printf("hello 71\n");
    Input[0] = t0;
    Output[0] = t2;

    TF_Tensor** InputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*) * NumInputs);
    TF_Tensor** OutputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*) * NumOutputs);
    InputValues[0] = input_tensor;
    printf("hello 78\n");
    TF_SessionRun(session, NULL, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, NULL, 0, NULL, status);
    check_status(status);
    printf("hello 81\n");
    // Clean up
    TF_DeleteTensor(input_tensor);
    TF_DeleteSession(session, status);
    TF_DeleteSessionOptions(session_opts);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);
    free(input_data);

    // ********* Retrieve and print the result
    void* buff = TF_TensorData(OutputValues[0]);
    float* offsets = (float*)buff;
    
    for (int i = 0; i < 10; i++) {
        printf("%f\n",offsets[i]);
    }
    return 0;
}

int main() {
    // Buffer size: one image (64x64x3) in float
    int size = NUM_IMAGES * IMAGE_SIZE * sizeof(float);
    
    // Allocate the buffer for the test data (uint32_t to match the function prototype)
    uint32_t *buf = (uint32_t*)malloc(size);
    if (buf == NULL) {
        fprintf(stderr, "Failed to allocate buffer\n");
        return -1;
    }

    // Fill the buffer with dummy data (just a simple pattern, e.g., incrementing values)
    for (int i = 0; i < size / sizeof(uint32_t); ++i) {
        buf[i] = i;  // This will store incrementing values, easily trackable
    }

    // Call the offrac_cnn function with the test data
    int result = offrac_cnn(buf, size, 0, 0);
    
    // Check if the function ran successfully
    if (result != 0) {
        printf("Function failed with error code: %d\n", result);
    } else {
        printf("Function executed successfully.\n");
    }

    // Clean up
    free(buf);
    return 0;
}