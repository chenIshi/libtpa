/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Author: Yixi Chen <yixi.chen@kaust.edu.sa>
 */

#ifndef _OFFRAC_H_
#define _OFFRAC_H_

/*
struct offrac_topk_params {
    short k;
}

// TODO: add more params
struct offrac_minmax_params {

}

// TODO: add more params
struct offrac_logit_params {

}

// params is a fixed (at most 60B) field customized by the accel operand
union accel_params {
    struct offrac_topk_params topk;
    struct offrac_minmax_params minmax;
    struct offrac_logit_params logit;
}

struct offrac_common_h {
    int conn_id;
    short accel;
    short size;
    union accel_params params;
}

// offrac session states
struct offrac_session_s {
    struct offrac_common_h common;
    bool enabled;
}
*/

#include <math.h>
#include <float.h>

#define MAX_BUF_SIZE 20480

// Enum for offrac supporting functions
typedef enum {
    TOPK = 1,
    MINMAX,
    LOGIT
} offrac_func_t;

// Offrac request handlers
typedef int (*offrac_func_ptr)(uint32_t *buf, int size, int offrac_size, int offrac_args);

int offrac_topk(uint32_t *buf, int size, int offrac_size, int offrac_args);
int offrac_minmax(uint32_t *buf, int size, int offrac_size, int offrac_args);
int offrac_logit(uint32_t *buf, int size, int offrac_size, int offrac_args);

// Handler interface for offrac functions
int offrac_process(char *buf, int size, offrac_func_t offrac_func, int offrac_size, int offrac_args);
void offrac_down(void);

#endif
