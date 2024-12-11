#include <stdio.h>

#define ENABLE_LOGGING
#include "logger.h"

void matmulkernel(void *devPtr, int rows_a, int cols_a, int cols_b, int num_threads)
{
    double *matA = (double *)(((void **)devPtr)[0]);
    double *matB = (double *)(((void **)devPtr)[1]);
    double *matC = (double *)(((void **)devPtr)[2]);

#pragma omp parallel for num_threads(num_threads) collapse(2)
    for (int i = 0; i < rows_a; i++)
    {
        for (int j = 0; j < cols_b; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < cols_a; k++)
            {
                sum += matA[i * cols_a + k] * matB[k * cols_b + j];
            }
            matC[i * cols_b + j] = sum;
        }
    }
}

extern void launchcpukernel(void *devPtr, int rows_a, int cols_a, int cols_b, int num_threads)
{
    log_message(LOG_DEBUG, "Launching CPU Kernel for Matrix Multiplication with %i rows (A), %i cols (A), %i cols (B), and %i threads.",
                rows_a, cols_a, cols_b, num_threads);
    matmulkernel(devPtr, rows_a, cols_a, cols_b, num_threads);
}
