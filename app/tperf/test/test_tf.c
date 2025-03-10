#include <stdio.h>
#include <stdlib.h>
#include "tensorflow/c/c_api.h"

// Error checking helper function
void check_status(TF_Status* status) {
    if (TF_GetCode(status) != TF_OK) {
        printf("Error: %s\n", TF_Message(status));
        exit(1);
    }
}

void NoOpDeallocator(void* data, size_t a, void* b) {}

int main() {
    //setenv("TF_CPP_MIN_LOG_LEVEL", "0", 1);
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
    float input_data[64 * 64 * 3] = {0};  // Replace with actual image data
    int64_t input_dims[] = {1, 64, 64, 3};  // Shape of the input image: (1, 64, 64, 3)
    // Create the input tensor
    TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT, input_dims, 4, input_data, sizeof(input_data), &NoOpDeallocator, NULL);
    
    // Prepare output tensor
    //****** Get input tensor
    int NumInputs = 1;
    TF_Output* Input = (TF_Output*)malloc(sizeof(TF_Output) * NumInputs);

    TF_Output t0 = {TF_GraphOperationByName(graph, "serving_default_input_1"), 0};
    if(t0.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName serving_default_input_1\n");
    else
	printf("TF_GraphOperationByName serving_default_input_1 is OK\n");
    
    Input[0] = t0;
    
    //********* Get Output tensor
    int NumOutputs = 1;
    TF_Output* Output = (TF_Output*)malloc(sizeof(TF_Output) * NumOutputs);

    TF_Output t2 = {TF_GraphOperationByName(graph, "StatefulPartitionedCall"), 0};
    if(t2.oper == NULL)
        printf("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n");
    else	
	printf("TF_GraphOperationByName StatefulPartitionedCall is OK\n");
    
    Output[0] = t2;

    TF_Tensor** InputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumInputs);
    TF_Tensor** OutputValues = (TF_Tensor**)malloc(sizeof(TF_Tensor*)*NumOutputs);
    InputValues[0] = input_tensor;

    // Run inference
    TF_SessionRun(session, NULL, Input, InputValues, 1, Output, OutputValues, 1, NULL, 0, NULL, status);
    check_status(status);

    // Clean up
    TF_DeleteTensor(input_tensor);
    TF_DeleteSession(session, status);
    TF_DeleteSessionOptions(session_opts);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);

    // ********* Retrieve and print the result
    void* buff = TF_TensorData(OutputValues[0]);
    float* offsets = (float*)buff;
    printf("Result Tensor :\n");
    for (int i = 0; i < 10; i++) {
        printf("%f\n",offsets[i]);
    }

    return 0;
}
