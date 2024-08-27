/*
Copyright 2023 Paweł Czarnul pczarnul@eti.pg.edu.pl

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#include "cudampi.h"
#include <cuda_runtime.h>
#include <mpi.h>

#define __CUDAMPI_MAX_THREAD_COUNT 1024 // maximum number of threads in the app (client/process 0 part)

extern __global__ void kernel(long *devPtr);

void __cudampi__setglobalpowerlimit(float powerlimit);
int __cudampi__selectdevicesforpowerlimit_greedy();

int __cudampi__getnextchunkindex(long long *globalcounter, int batchsize, long long max);
int __cudampi__getnextchunkindex_enableddevices(long long *globalcounter, int batchsize, long long max);
int __cudampi__getnextchunkindex_alldevices(long long *globalcounter, int batchsize, long long max);

void __cudampi__initializeMPI(int argc, char **argv);

void __cudampi__terminateMPI();

int __cudampi__gettargetGPU(int device);

int __cudampi__gettargetMPIrank(int device);

cudaError_t __cudampi__cudaMalloc(void **devPtr, size_t size);

cudaError_t __cudampi__cudaFree(void *devPtr);

cudaError_t __cudampi__cudaDeviceSynchronize(void);

cudaError_t __cudampi__cudaSetDevice(int device);

cudaError_t __cudampi__cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind);

cudaError_t __cudampi__cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);

void __cudampi__kernel(void *devPtr);

void __cudampi__kernelinstream(void *devPtr, cudaStream_t stream);

cudaError_t __cudampi__cudaGetDeviceCount(int *count);

void __cudampi__getCUDAdevicescount(int *cudadevicescount);

cudaError_t __cudampi__cudaStreamCreate(cudaStream_t *pStream);

cudaError_t __cudampi__cudaStreamDestroy(cudaStream_t stream);
