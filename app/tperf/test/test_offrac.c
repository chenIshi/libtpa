#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include <inttypes.h>

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

void print_first_5_elements(void *buf, int size, int is_float) {
    union {
        uint32_t *int_buf;
        float *float_buf;
    } u;
    u.int_buf = (uint32_t *)buf;

    printf("First 5 elements: ");
    for (int i = 0; i < 5 && i < size; i++) {
        if (is_float) {
            printf("%f ", u.float_buf[i]);
        } else {
            printf("%" PRIu32 " ", u.int_buf[i]);
        }
    }
    printf("\n");
}

void test_offrac_topk() {
    uint32_t buf[] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
    int size = sizeof(buf) / sizeof(buf[0]);
    int k = 5;

    printf("Before offrac_topk:\n");
    print_first_5_elements(buf, size, 0);

    offrac_topk(buf, size, 0, k);

    printf("After offrac_topk:\n");
    print_first_5_elements(buf, size, 0);
}

void test_offrac_minmax() {
    float buf[] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0};
    int size = sizeof(buf) / sizeof(buf[0]);

    printf("Before offrac_minmax:\n");
    print_first_5_elements(buf, size, 1);

    offrac_minmax(buf, size, 0, 0);

    printf("After offrac_minmax:\n");
    print_first_5_elements(buf, size, 1);
}

void test_offrac_logit() {
    float buf[] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0};
    int size = sizeof(buf) / sizeof(buf[0]);

    printf("Before offrac_logit:\n");
    print_first_5_elements(buf, size, 1);

    offrac_logit(buf, size, 0, 0);

    printf("After offrac_logit:\n");
    print_first_5_elements(buf, size, 1);
}

int main() {
    test_offrac_topk();
    test_offrac_minmax();
    test_offrac_logit();
    return 0;
}