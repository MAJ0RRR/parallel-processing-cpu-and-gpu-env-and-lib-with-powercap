#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define ENABLE_LOGGING_GPU
#define ENABLE_LOGGING
#include "logger_gpu.h"
#include "logger.h"
#include "matmul_defines.h"

__global__ void matmulkernel(void *devPtr)
{
    double *matrixA = (double *)(((void **)devPtr)[0]);
    double *matrixB = (double *)(((void **)devPtr)[1]);
    double *matrixC = (double *)(((void **)devPtr)[2]);

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < MATMUL_MATRIX_SIZE && col < MATMUL_MATRIX_SIZE)
    {
        double sum = 0.0;
        for (int k = 0; k < MATMUL_MATRIX_SIZE; k++)
        {
            sum += matrixA[row * MATMUL_MATRIX_SIZE + k] * matrixB[k * MATMUL_MATRIX_SIZE + col];
        }
        matrixC[row * MATMUL_MATRIX_SIZE + col] = sum;
    }
}

extern "C" void launchkernelinstream(void *devPtr, unsigned long batchSize, cudaStream_t stream)
{
    dim3 threadsPerBlock(MATMUL_THREADS_IN_BLOCK, MATMUL_THREADS_IN_BLOCK);
    dim3 blocksPerGrid(batchSize / MATMUL_THREADS_IN_BLOCK, MATMUL_MATRIX_SIZE / MATMUL_THREADS_IN_BLOCK);

    matmulkernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(devPtr);

    if (cudaSuccess != cudaGetLastError())
    {
        log_message(LOG_ERROR, "Error during kernel launch in stream");
    }
}

extern "C" void launchkernel(void *devPtr, unsigned long batchSize) { launchkernelinstream(devPtr, batchSize, 0); }
