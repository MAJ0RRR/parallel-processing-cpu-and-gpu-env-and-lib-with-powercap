#include <stdio.h>
#include "mkl.h"

#define ENABLE_LOGGING
#include "logger.h"

void matmulkernel(void *devPtr, int rows_a, int cols_a, int cols_b, int num_threads)
{
    double *matA = (double *)(((void **)devPtr)[0]);
    double *matB = (double *)(((void **)devPtr)[1]);
    double *matC = (double *)(((void **)devPtr)[2]);

    double alpha = 1.0;
    double beta = 0.0;

    mkl_set_num_threads(num_threads);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                rows_a, cols_b, cols_a, 
                alpha, matA, cols_a, matB, cols_b, 
                beta, matC, cols_b);
}

extern void launchcpukernel(void *devPtr, int rows_a, int cols_a, int cols_b, int num_threads)
{
    log_message(LOG_DEBUG, "Launching CPU Kernel for Matrix Multiplication with %i rows (A), %i cols (A), %i cols (B), and %i threads.",
                rows_a, cols_a, cols_b, num_threads);
    matmulkernel(devPtr, rows_a, cols_a, cols_b, num_threads);
}
