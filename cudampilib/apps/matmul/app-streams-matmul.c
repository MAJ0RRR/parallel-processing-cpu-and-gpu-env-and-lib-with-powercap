#include "cudampilib.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <assert.h>

#define ENABLE_LOGGING
#include "logger.h"
#include "matmul_defines.h"

#define ENABLE_OUTPUT_LOGS
#include "utility.h"

struct __cudampi__arguments_type __cudampi__arguments;

long long MATRIX_SIZE;

double *matrixA;
double *matrixB;
double *matrixC;

unsigned long batchSize;
long long globalCounter = 0;

int streamCount = 1;

int main(int argc, char **argv)
{
    struct timeval start, stop;
    struct timeval startTotal, stopTotal;

    gettimeofday(&startTotal, NULL);

    __cudampi__initializeMPI(argc, argv);

    streamCount = __cudampi__arguments.number_of_streams;
    batchSize = __cudampi__arguments.batch_size;
    MATRIX_SIZE = __cudampi__arguments.problem_size;

    assert(batchSize % MATMUL_THREADS_IN_BLOCK == 0);

    int allDevicesCount = 0;

    __cudampi__getDeviceCount(&allDevicesCount);

    // Przydzielanie pamięci hosta
    cudaHostAlloc((void **)&matrixA, sizeof(double) * MATRIX_SIZE * MATRIX_SIZE, cudaHostAllocDefault);
    cudaHostAlloc((void **)&matrixB, sizeof(double) * MATRIX_SIZE * MATRIX_SIZE, cudaHostAllocDefault);
    cudaHostAlloc((void **)&matrixC, sizeof(double) * MATRIX_SIZE * MATRIX_SIZE, cudaHostAllocDefault);

    // Inicjalizacja danych wejściowych
    for (size_t i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++)
    {
        matrixA[i] = ((int)i % 100) * 0.01;
        matrixB[i] = ((int)i % 50) * 0.02;
        matrixC[i] = 0.0;
    }

    gettimeofday(&start, NULL);

#pragma omp parallel num_threads(allDevicesCount)
    {
        // Wątki OpenMP zajmują się różnymi urządzeniami
        __cudampi__batch_pointer batchPointer;
        int finish = 0;
        int myThreadID = omp_get_thread_num();
        void *devMatrixA, *devMatrixB, *devMatrixC;
        void *devPointer;
        cudaStream_t stream;

        __cudampi__setDevice(myThreadID);
        __cudampi__malloc(&devMatrixA, batchSize * MATRIX_SIZE * sizeof(double));
        __cudampi__malloc(&devMatrixB, MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
        __cudampi__malloc(&devMatrixC, batchSize * MATRIX_SIZE * sizeof(double));
        __cudampi__malloc(&devPointer, 3 * sizeof(void *));

        __cudampi__memcpyAsync(devMatrixB, matrixB, MATRIX_SIZE * MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice, 0);

        __cudampi__streamCreate(&stream);

        do
        {
            batchPointer = __cudampi__getnextchunkindex(&globalCounter, MATRIX_SIZE);
            if (batchPointer.start >= MATRIX_SIZE)
            {
                finish = 1;
            }
            else
            {
                size_t rows = batchPointer.n_elements;

                __cudampi__memcpyAsync(devMatrixA, matrixA + batchPointer.start * MATRIX_SIZE, rows * MATRIX_SIZE * sizeof(double), cudaMemcpyHostToDevice, stream);
                __cudampi__kernelInStream(devPointer, stream);
                __cudampi__memcpyAsync(matrixC + batchPointer.start * MATRIX_SIZE, devMatrixC, rows * MATRIX_SIZE * sizeof(double), cudaMemcpyDeviceToHost, stream);
            }
        } while (!finish);

        __cudampi__streamDestroy(stream);
        __cudampi__free(devMatrixA);
        __cudampi__free(devMatrixB);
        __cudampi__free(devMatrixC);
        __cudampi__free(devPointer);
    }

    gettimeofday(&stop, NULL);
    log_message(LOG_INFO, "Main elapsed time=%f\n", (double)((stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) / 1000000.0));

    __cudampi__terminateMPI();

    cudaFreeHost(matrixA);
    cudaFreeHost(matrixB);
    cudaFreeHost(matrixC);

    gettimeofday(&stopTotal, NULL);
    log_message(LOG_INFO, "Total elapsed time=%f\n", (double)((stopTotal.tv_sec - startTotal.tv_sec) + (double)(stopTotal.tv_usec - startTotal.tv_usec) / 1000000.0));

    return 0;
}
