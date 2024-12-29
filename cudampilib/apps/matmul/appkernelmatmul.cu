#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define ENABLE_LOGGING_GPU
#define ENABLE_LOGGING
#include "logger_gpu.h"
#include "logger.h"
#include "matmul_defines.h"

#define TILE_SIZE 32

#include <cublas_v2.h>

void matmulCublas(void *devPtr) {
    log_message(LOG_INFO, "This works");

    log_message(LOG_INFO, "devPtr pointer: %p", devPtr);

    void** new_ptr[3];
    cudaMemcpy(new_ptr, devPtr, 3*sizeof(void*), cudaMemcpyDeviceToHost);

    // log_message(LOG_INFO, "matrixA pointer: %p", (((void **)devPtr)[0]));
    // log_message(LOG_INFO, "matrixB pointer: %p", (((void **)devPtr)[1]));
    // log_message(LOG_INFO, "matrixC pointer: %p", (((void **)devPtr)[2]));

    double *matrixA = (double *)(((void **)new_ptr)[0]);
    double *matrixB = (double *)(((void **)new_ptr)[1]);
    double *matrixC = (double *)(((void **)new_ptr)[2]);

    log_message(LOG_INFO, "After matrices mem separation");

    cublasHandle_t handle;
    cublasCreate(&handle);

    double alpha = 1.0;
    double beta = 0.0;

    log_message(LOG_INFO, "Multiplication");

    // Wykonaj mnożenie macierzy A * B = C za pomocą cuBLAS
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, MATMUL_MATRIX_SIZE, MATMUL_MATRIX_SIZE, MATMUL_MATRIX_SIZE,
                &alpha, matrixA, MATMUL_MATRIX_SIZE, matrixB, MATMUL_MATRIX_SIZE,
                &beta, matrixC, MATMUL_MATRIX_SIZE);

    log_message(LOG_INFO, "After multiplication");

    cublasDestroy(handle);
}

__global__ void matmulkernel(void *devPtr)
{
    double *matrixA = (double *)(((void **)devPtr)[0]);
    double *matrixB = (double *)(((void **)devPtr)[1]);
    double *matrixC = (double *)(((void **)devPtr)[2]);

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ double tileA[TILE_SIZE][TILE_SIZE];
    __shared__ double tileB[TILE_SIZE][TILE_SIZE];

    double sum = 0.0;

    for (int m = 0; m < (MATMUL_MATRIX_SIZE  / TILE_SIZE); ++m) {
        tileA[threadIdx.y][threadIdx.x] = matrixA[row * MATMUL_MATRIX_SIZE + (m * TILE_SIZE + threadIdx.x)];
        tileB[threadIdx.y][threadIdx.x] = matrixB[(m * TILE_SIZE + threadIdx.y) * MATMUL_MATRIX_SIZE  + col];

        __syncthreads();  

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < MATMUL_MATRIX_SIZE  && col < MATMUL_MATRIX_SIZE) {
        matrixC[row * MATMUL_MATRIX_SIZE + col] = sum;
    }
}

extern "C" void launchkernelinstream(void *devPtr, unsigned long batchSize, cudaStream_t stream)
{
    dim3 threadsPerBlock(MATMUL_THREADS_IN_BLOCK, MATMUL_THREADS_IN_BLOCK);
    dim3 blocksPerGrid(batchSize / MATMUL_THREADS_IN_BLOCK, MATMUL_MATRIX_SIZE / MATMUL_THREADS_IN_BLOCK);

    // matmulkernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(devPtr);
    matmulCublas(devPtr);

    if (cudaSuccess != cudaGetLastError())
    {
        log_message(LOG_ERROR, "Error during kernel launch in stream");
    }
}

extern "C" void launchkernel(void *devPtr, unsigned long batchSize) { launchkernelinstream(devPtr, batchSize, 0); }
