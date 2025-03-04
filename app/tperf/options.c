/*
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2021-2023, ByteDance Ltd. and/or its Affiliates
 * Author: Yuanhan Liu <liuyuanhan.131@bytedance.com>
 */
#include <stdio.h>

#include <utils.h>

#include "tperf.h"

void usage(void)
{
	fprintf(stderr, "usage: tperf [options]\n"
			"\n"
			"       tperf -s [options]\n"
			"       tperf -t test [options]\n"
			"\n"
			"Tperf, a tpa performance benchmark.\n"
			"\n"
			"Client options:\n"
			"  -c server         run in client mode (the default mode) and specifies the server \n"
			"                    address (default: 127.0.0.1)\n"
			"  -t test           specifies the test mode, which is listed below\n"
			"  -p port           specifies the port to connect to (default: %d)\n"
			"  -d duration       specifies the test duration (default: 10s)\n"
			"  -m message_size   specifies the message size (default: %d)\n"
			"  -n nr_thread      specifies the thread count (default: 1)\n"
			"  -i                do integrity verification (default: off)\n"
			"  -C nr_conn        specifies the connection to be created for each thread (default: 1)\n"
			"  -W 0|1            disable/enable zero copy write (default: on)\n"
			"  -S start_cpu      specifies the starting cpu to bind\n"
			"\n"
			"Server options:\n"
			"  -s                run in server mode\n"
			"  -n nr_thread      specifies the thread count (default: 1)\n"
			"  -l addr           specifies local address to listen on\n"
			"  -p port           specifies the port to listen on (default: %d)\n"
			"  -S start_cpu      specifies the starting cpu to bind\n"
			"\n"
			"The supported test modes are:\n"
			"  * read            read data from the server end\n"
			"  * write           write data to the server end\n"
			"  * rw              test read and write simultaneously\n"
			"  * rr              send a request (with payload) to the server and\n"
			"                    expects a response will be returned from the server end\n"
			"  * crr             basically does the same thing like rr, except that a\n"
			"                    connection is created for each request\n"
			"Offrac options (both options need to be set the same time, exclusive to -i):\n"
			"  -f offrac_function specifies the offrac function to be performed\n"
			"  -z offrac_size     specifies the value array size within a request\n"
			"  -a offrac_args	  specifies the optional args for several offrac functions\n", 
			TPERF_PORT,
			DEFAULT_MESSAGE_SIZE,
			TPERF_PORT
	);

	exit(1);
}

#define PARSE_NUM(var, optarg, type, name)	do {			\
	var = tpa_parse_num(optarg, type);				\
	if (errno) {							\
		fprintf(stderr, "invalid %s: %s\n", name, optarg);	\
		exit(1);						\
	}								\
} while (0)

int parse_options(int argc, char **argv)
{
	int opt;
	int has_opt_c = 0;

	memset(&ctx, 0, sizeof(ctx));

	ctx.is_client     = 1;
	ctx.server        = "127.0.0.1";
	ctx.test          = -1;
	ctx.port          = TPERF_PORT;
	ctx.nr_thread     = DEFAULT_NR_THREAD;
	ctx.duration      = DEFAULT_DURATION;
	ctx.message_size  = DEFAULT_MESSAGE_SIZE;
	ctx.enable_tso    = 1;
	ctx.offrac_function = 0;
	ctx.offrac_size = 0;
	ctx.offrac_args = 0;
	ctx.enable_zwrite = 1;
	ctx.start_cpu     = -4096;
	ctx.nr_conn_per_thread = 1;

	while ((opt = getopt(argc, argv, "c:C:t:d:l:m:n:p:S:W:f:z:a:isqh")) != -1) {
		switch (opt) {
		case 's':
			ctx.is_client = 0;
			break;

		case 'c':
			ctx.server = strdup(optarg);
			has_opt_c = 1;
			break;

		case 'C':
			PARSE_NUM(ctx.nr_conn_per_thread, optarg, NUM_TYPE_NONE, "connection count");
			break;

		case 't':
			ctx.test = str_to_test(optarg);
			if (ctx.test < 0) {
				fprintf(stderr, "invalid test mode: %s\n", optarg);
				exit(1);
			}
			break;

		case 'T':
			PARSE_NUM(ctx.enable_tso, optarg, NUM_TYPE_NONE, "tso enabling");
			break;

		case 'd':
			PARSE_NUM(ctx.duration, optarg, NUM_TYPE_TIME, "duration");
			break;

		case 'l':
			ctx.local = strdup(optarg);
			break;

		case 'm':
			PARSE_NUM(ctx.message_size, optarg, NUM_TYPE_SIZE, "message size");
			break;

		case 'n':
			PARSE_NUM(ctx.nr_thread, optarg, NUM_TYPE_NONE, "thread count");
			break;

		case 'p':
			PARSE_NUM(ctx.port, optarg, NUM_TYPE_NONE, "port");
			if (ctx.port <= 0 || ctx.port >= 65536) {
				fprintf(stderr, "invalid port: %d: out of range\n", ctx.port);
				exit(1);
			}
			break;

		case 'W':
			PARSE_NUM(ctx.enable_zwrite, optarg, NUM_TYPE_NONE, "zwrite enabling");
			break;

		case 'S':
			PARSE_NUM(ctx.start_cpu, optarg, NUM_TYPE_NONE, "start_cpu");
			break;
		
		case 'f':
			PARSE_NUM(ctx.offrac_function, optarg, NUM_TYPE_NONE, "offrac function");
			if (ctx.offrac_function < 0 || ctx.offrac_function >= 4) {
				fprintf(stderr, "invalid offrac function: %d: out of range\n", ctx.offrac_function);
				exit(1);
			}
			break;

		case 'a':
			PARSE_NUM(ctx.offrac_args, optarg, NUM_TYPE_NONE, "offrac func args");
			break;

		case 'z':
			PARSE_NUM(ctx.offrac_size, optarg, NUM_TYPE_NONE, "offrac size");
			break;

		case 'i':
			ctx.integrity_enabled = 1;
			break;

		case 'q':
			ctx.quiet = 1;
			break;

		case 'h':
			usage();
			break;

		default:
			usage();
		}
	}

	if (has_opt_c && ctx.is_client == 0) {
		fprintf(stderr, "error: -c and -s can not be given at the same time\n\n");
		usage();
	}

	if (ctx.is_client && ctx.test < 0) {
		fprintf(stderr, "error: missing mandatory option: -t test\n\n");
		usage();
	}

	if (ctx.integrity_enabled && ctx.offrac_function) {
		fprintf(stderr, "error: -i and -f offrac_function can not be given at the same time\n\n");
		usage();
	}

	if (ctx.offrac_function && ctx.offrac_size == 0) {
		fprintf(stderr, "error: missing mandatory option: -s offrac size\n\n");
		usage();
	}

	if (ctx.offrac_size && ctx.offrac_function == 0) {
		fprintf(stderr, "error: missing mandatory option: -f offrac function\n\n");
		usage();
	}

	if (ctx.offrac_args && ctx.offrac_function == 0) {
		fprintf(stderr, "error: -a offrac args can only be used with -f offrac function\n\n");
		usage();
	}

	return 0;
}
