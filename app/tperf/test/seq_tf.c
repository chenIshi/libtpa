#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <float.h>
#include <inttypes.h>

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

    // Prepare input tensor dimensions for a single image (not batch)
    int64_t input_dims[] = {1, 64, 64, 3};  // (1 image, 64x64, RGB)

    // Get input and output tensor names
    TF_Output input_op = {TF_GraphOperationByName(graph, "serving_default_input_1"), 0};
    TF_Output output_op = {TF_GraphOperationByName(graph, "StatefulPartitionedCall"), 0};

    if (input_op.oper == NULL || output_op.oper == NULL) {
        fprintf(stderr, "Failed to get input/output tensor\n");
        return -1;
    }

    int num_images = size / (IMAGE_SIZE * sizeof(float));
    float* input_data = (float *)malloc(IMAGE_SIZE * sizeof(float));
    if (input_data == NULL) {
        fprintf(stderr, "Failed to allocate input data\n");
        return -1;
    }

    for (int i = 0; i < num_images; i++) {
        // Copy one image from the buffer
        memcpy(input_data, buf + i * IMAGE_SIZE, IMAGE_SIZE * sizeof(float));

        // Create an input tensor for this single image
        TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT, input_dims, 4, input_data, IMAGE_SIZE * sizeof(float), &NoOpDeallocator, NULL);

        // Run inference
        TF_Tensor* output_tensor = NULL;
        TF_SessionRun(session, NULL, &input_op, &input_tensor, 1, &output_op, &output_tensor, 1, NULL, 0, NULL, status);
        check_status(status);

        // Retrieve and print the result for this image
        void* buff = TF_TensorData(output_tensor);
        float* offsets = (float*)buff;

    /*
	printf("Image %d result:\n", i);
        for (int j = 0; j < 10; j++) {
            printf("%f\n", offsets[j]);
        }
	*/
	// Store the result back in buf (in-place update)
	// TODO: update to output
        memcpy(buf + i * 10, offsets, 10 * sizeof(float));

        // Clean up for this image
        TF_DeleteTensor(input_tensor);
        TF_DeleteTensor(output_tensor);
    }

    // Clean up global resources
    TF_DeleteSession(session, status);
    TF_DeleteSessionOptions(session_opts);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);
    free(input_data);

    return num_images * 10 * sizeof(float);
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
    /*
    if (result != 0) {
        printf("Function failed with error code: %d\n", result);
    } else {
        printf("Function executed successfully.\n");
    }
    */

    // Clean up
    free(buf);
    return 0;
}
