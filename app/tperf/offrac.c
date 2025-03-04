/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Author: Yixi Chen <yixi.chen@kaust.edu.sa>
 */

#include "tperf.h"

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
    uint32_t *buf32 = (uint32_t *)buf;
    int size32 = size / sizeof(uint32_t);

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
        default:
            fprintf(stderr, "failed to recognize OffRAC function %d\n", offrac_func);
            return -1;
    }

    if (offrac_func_ptr) {
        return offrac_func_ptr(buf32, size32, offrac_size, offrac_args);
    } else {
	fprintf(stderr, "Reaching unreachable region in offrac processing\n");
	return -1;
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
