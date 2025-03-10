/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Author: Yixi Chen <yixi.chen@kaust.edu.sa>
 */

#include "tperf.h"

static int call_counter = 0;
static char *buffer_storage = NULL;
static int current_buffer_size = 0;

int offrac_process(char *buf, int size, offrac_func_t offrac_func, int offrac_size, int offrac_args) { 
    // slice the byte array into uint32_t array
    // Check if the buffer is properly aligned
    if ((uintptr_t)buf % sizeof(uint32_t) != 0) {
        fprintf(stderr, "Buffer is not properly aligned for uint32_t access\n");
        return -1;
    }

    // Calculate the number of 32-bit integers
    if (size % sizeof(uint32_t) != 0) {
        fprintf(stderr, "Buffer size is not a multiple of uint32_t size\n");
        return -1;
    }

    // Allocate storage buffer if not already allocated
    if (buffer_storage == NULL) {
        buffer_storage = (char *)malloc(MAX_BUF_SIZE);
        if (buffer_storage == NULL) {
            fprintf(stderr, "Failed to allocate buffer storage\n");
            return -1;
        }
    }

    // Check if there is enough space in the buffer
    if (call_counter * size >= MAX_BUF_SIZE) {
        fprintf(stderr, "Buffer storage is full\n");
        return -1;
    }

    // Copy the current buffer to the storage buffer
    memcpy(buffer_storage + (call_counter * size), buf, size);
    current_buffer_size += size;
    call_counter ++;

    // Check if we have reached the required number of calls
    if (call_counter < offrac_size) {
        return 0;
    }

    // Process the buffer once the data in request all arrived

    uint32_t *buf32 = (uint32_t *)buffer_storage;
    int size32 = current_buffer_size / sizeof(uint32_t);

    offrac_func_ptr offrac_func_ptr = NULL;
    switch (offrac_func) {
        case TOPK:
            offrac_func_ptr = offrac_topk;
            break;
        case MINMAX:
            offrac_func_ptr = offrac_minmax;
            break;
        case LOGIT:
            offrac_func_ptr = offrac_logit;
            break;
        case CNN:
            offrac_func_ptr = offrac_cnn;
            break;
        default:
            fprintf(stderr, "failed to recognize OffRAC function %d\n", offrac_func);
            return -1;
    }

    if (offrac_func_ptr) {
        if (offrac_func_ptr(buf32, size32, offrac_size, offrac_args) >= 0) {
            // Reset the call counter and current buffer size
            call_counter = 0;
            current_buffer_size = 0;
            return 0;
        } else {
            fprintf(stderr, "failed during running offrac function\n");
            return -1;
        }
    } else {
        fprintf(stderr, "reaching unreachable region in offrac processing\n");
        return -1;
    }
}

void offrac_down() {
    // free buffer storage
    if (buffer_storage) {
        free(buffer_storage);
        buffer_storage = NULL;
    }
}

// Comparison function for qsort (descending order)
int compare_desc(const void *a, const void *b) {
    return (*(uint32_t *)b - *(uint32_t *)a);
}

// return the buf size after topk, ideally should be k
// return -1 if error
int offrac_topk(uint32_t *buf, int size, int offrac_size, int offrac_args) {
    int k = offrac_args;

    // Check if k is valid
    if (k <= 0 || k > size) {
        fprintf(stderr, "Invalid value of k: %d\n", k);
        return -1;
    }

    // Sort the array in descending order
    qsort(buf, size, sizeof(uint32_t), compare_desc);

    // Set elements after the k-th element to 0
    for (int i = k; i < size; i++) {
        buf[i] = 0;
    }

    return k;
}

// return the buf size after minmax norm, ideally should remains the same
// return -1 if error
int offrac_minmax(uint32_t *buf, int size, int offrac_size, int offrac_args) {
    float *float_buf = (float *)buf;

    // Find the min and max values
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;
    for (int i = 0; i < size; i++) {
        if (float_buf[i] < min_val) {
            min_val = float_buf[i];
        }
        if (float_buf[i] > max_val) {
            max_val = float_buf[i];
        }
    }

    // Check if min and max values are the same
    if (min_val == max_val) {
        fprintf(stderr, "All elements are the same\n");
        return -1;
    }

    // Normalize the elements using min-max normalization
    for (int i = 0; i < size; i++) {
        float_buf[i] = (float_buf[i] - min_val) / (max_val - min_val);
    }

    return size;
}

// return the buf size after logit transformation, ideally should remains the same
// return -1 if error
int offrac_logit(uint32_t *buf, int size, int offrac_size, int offrac_args) {
    float *float_buf = (float *)buf;

    // Apply the logistic function to each element
    for (int i = 0; i < size; i++) {
        float_buf[i] = 1.0f / (1.0f + expf(-float_buf[i]));
    }

    return size;
}

// return the buf size after cnn, ideally should be the 10 floating point value (one-hot encoding)
// return -1 if error
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
