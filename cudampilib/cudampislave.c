/*
Copyright 2023 Paweł Czarnul pczarnul@eti.pg.edu.pl

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "cudampi.h"
#include "cudampicommon.h"

int __cudampi__MPIproccount;
int __cudampi__myrank;

float lastEnergyMeasured = 0.0;

int *__cudampi_targetMPIrankfordevice; // MPI rank for device number (global)
int *__cudampi__GPUcountspernode;
int *__cudampi__freeThreadsPerNode;

MPI_Comm *__cudampi__communicators;

int __cudampi_totaldevicecount = 0; // how many GPUs in total (on all considered nodes)
int __cudampi__localGpuDeviceCount = 1;
int __cudampi__localFreeThreadCount = 0;

void launchkernel(void *devPtr);
void launchkernelinstream(void *devPtr, cudaStream_t stream);
void launchcpukernel(void *devPtr, int thread_count);

int main(int argc, char **argv) {

  // basically this is a slave process that waits for requests and redirects
  // those to local GPU(s)

  int mtsprovided;

  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mtsprovided);

  if (mtsprovided != MPI_THREAD_MULTIPLE) {
    printf("\nNo support for MPI_THREAD_MULTIPLE mode.\n");
    fflush(stdout);
    exit(-1);
  }
  
  // enable nested omp parallels
  omp_set_nested(1);

  // fetch information about the rank and number of processes

  MPI_Comm_size(MPI_COMM_WORLD, &__cudampi__MPIproccount);
  MPI_Comm_rank(MPI_COMM_WORLD, &__cudampi__myrank);

  // now check the number of GPUs available and synchronize with process 0 that wants this info

  if (cudaSuccess != cudaGetDeviceCount(&__cudampi__localGpuDeviceCount)) {
    printf("Error invoking cudaGetDeviceCount()");
    fflush(stdout);
    exit(-1);
  }

  __cudampi__GPUcountspernode = (int *)malloc(sizeof(int) * __cudampi__MPIproccount);
  if (!__cudampi__GPUcountspernode) {
    printf("\nNot enough memory");
    exit(-1); // we could exit in a nicer way! TBD
  }

  __cudampi__freeThreadsPerNode = (int *)malloc(sizeof(int) * __cudampi__MPIproccount);
  if (!__cudampi__freeThreadsPerNode) {
    printf("\nNot enough memory");
    exit(-1); // we could exit in a nicer way! TBD
  }

  if (cudaSuccess != __cudampi__getCpuFreeThreads(&__cudampi__localFreeThreadCount)) {
    printf("Error invoking __cudampi__getCpuFreeThreads()");
    fflush(stdout);
    exit(-1);
  }

  MPI_Allgather(&__cudampi__localGpuDeviceCount, 1, MPI_INT, __cudampi__GPUcountspernode, 1, MPI_INT, MPI_COMM_WORLD);

  MPI_Allgather(&__cudampi__localFreeThreadCount, 1, MPI_INT, __cudampi__freeThreadsPerNode, 1, MPI_INT, MPI_COMM_WORLD);
  
  MPI_Bcast(&__cudampi_totaldevicecount, 1, MPI_INT, 0, MPI_COMM_WORLD);

  __cudampi_targetMPIrankfordevice = (int *)malloc(__cudampi_totaldevicecount * sizeof(int));
  if (!__cudampi_targetMPIrankfordevice) {
    printf("\nNot enough memory");
    exit(-1); // we could exit in a nicer way! TBD
  }

  MPI_Bcast(__cudampi_targetMPIrankfordevice, __cudampi_totaldevicecount, MPI_INT, 0, MPI_COMM_WORLD);

  // create communicators
  // in the case of the slave we need to go by every GPU and for each GPU there will be a separate GPU shared with the master -- process 0

  MPI_Comm *__cudampi__communicators = (MPI_Comm *)malloc(sizeof(MPI_Comm) * __cudampi_totaldevicecount);
  if (!__cudampi__communicators) {
    printf("\nNot enough memory for communicators");
    exit(-1); // we could exit in a nicer way! TBD
  }

  int commcounter = 0;

  for (int i = __cudampi__GPUcountspernode[0]; i < __cudampi_totaldevicecount; i++) {

    int ranks[2] = {0, __cudampi_targetMPIrankfordevice[i]}; // group and communicator between process 0 and the process of the target GPU/device

    MPI_Group groupall;
    MPI_Comm_group(MPI_COMM_WORLD, &groupall);

    // Keep only process 0 and the process handling the given GPU
    MPI_Group tempgroup;
    MPI_Group_incl(groupall, 2, ranks, &tempgroup);

    MPI_Comm tempcomm;
    if (__cudampi_targetMPIrankfordevice[i] != __cudampi__myrank) {
      MPI_Comm_create(MPI_COMM_WORLD, tempgroup, &tempcomm);
    } else {
      MPI_Comm_create(MPI_COMM_WORLD, tempgroup, &(__cudampi__communicators[commcounter++]));
    }
  }

  // here we need to spawn threads -- each responsible for handling one local GPU
  // spawn one thread for CPU processing if there are free cores, 0 otherwise
  int numberOfCpuThreads = __cudampi__localFreeThreadCount > 0;
  #pragma omp parallel num_threads(__cudampi__localGpuDeviceCount + numberOfCpuThreads)
  {

    MPI_Status status;

    // following communication needs to use dedicated communicators, not MPI_COMM_WORLD!

    do {
      MPI_Probe(0, MPI_ANY_TAG, __cudampi__communicators[omp_get_thread_num()], &status);

      if (status.MPI_TAG == __cudampi__CUDAMPIMALLOCREQ) {

        unsigned long rdata;

        MPI_Recv((unsigned long *)(&rdata), 1, MPI_UNSIGNED_LONG, 0, __cudampi__CUDAMPIMALLOCREQ, __cudampi__communicators[omp_get_thread_num()], &status);

        // allocate memory on the current GPU
        void *devPtr;

        cudaError_t e = cudaMalloc(&devPtr, (size_t)rdata);

        int ssize = sizeof(void *) + sizeof(cudaError_t);
        // send confirmation with the actual pointer
        unsigned char sdata[ssize];

        *((void **)sdata) = devPtr;
        *((cudaError_t *)(sdata + sizeof(void *))) = e;

        MPI_Send(sdata, ssize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPIMALLOCRESP, __cudampi__communicators[omp_get_thread_num()]);
      }

      if (status.MPI_TAG == __cpumpi__CPUMALLOCREQ) {

        unsigned long rdata;

        MPI_Recv((unsigned long *)(&rdata), 1, MPI_UNSIGNED_LONG, 0, __cpumpi__CPUMALLOCREQ, __cudampi__communicators[omp_get_thread_num()], &status);

        // allocate memory on the current GPU
        void *devPtr = = malloc((size_t)rdata);

        int ssize = sizeof(void *) + sizeof(cudaError_t);
        // send confirmation with the actual pointer
        unsigned char sdata[ssize];

        *((void **)sdata) = devPtr;

        // return cudaSuccess if memory is not NULL, cudaErrorMemoryAllocation otherwise
        cudaError_t e = (devPtr == NULL) ? cudaErrorMemoryAllocation : cudaSuccess;
        *((cudaError_t *)(sdata + sizeof(void *))) = e;

        MPI_Send(sdata, ssize, MPI_UNSIGNED_CHAR, 0, __cpumpi__CPUMALLOCRESP, __cudampi__communicators[omp_get_thread_num()]);
      }

      if (status.MPI_TAG == __cudampi__CUDAMPIFREEREQ) {

        int rsize = sizeof(void *);
        unsigned char rdata[rsize];

        MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPIFREEREQ, __cudampi__communicators[omp_get_thread_num()], &status);

        void *devPtr = *((void **)rdata);

        cudaError_t e = cudaFree(devPtr);

        MPI_Send((unsigned char *)(&e), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPIFREERESP, __cudampi__communicators[omp_get_thread_num()]);
      }

      if (status.MPI_TAG == __cpumpi__CPUFREEREQ) {

        int rsize = sizeof(void *);
        unsigned char rdata[rsize];

        MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cpumpi__CPUFREEREQ, __cudampi__communicators[omp_get_thread_num()], &status);

        void *devPtr = *((void **)rdata);
        cudaError_t e = cudaErrorInvalidDevicePointer;

        // maybe we should somehow handle invalid devPtr ?
        if (devPtr != NULL) {
          free(devPtr);
          e = cudaSuccess;
        }

        MPI_Send((unsigned char *)(&e), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 0, __cpumpi__CPUFREERESP, __cudampi__communicators[omp_get_thread_num()]);
      }

      if (status.MPI_TAG == __cudampi__CUDAMPIDEVICESYNCHRONIZEREQ) {

        int measurepower;

        MPI_Recv(&measurepower, 1, MPI_INT, 0, __cudampi__CUDAMPIDEVICESYNCHRONIZEREQ, __cudampi__communicators[omp_get_thread_num()], &status);

        // perform power measurement and attach it to the response

        size_t ssize = sizeof(cudaError_t) + sizeof(float);
        unsigned char sdata[ssize];

        int device;
        cudaGetDevice(&device);

        cudaError_t e = cudaDeviceSynchronize();
        *((cudaError_t *)sdata) = e;

        *((float *)(sdata + sizeof(cudaError_t))) = (measurepower ? getGPUpower(device) : -1); // -1 if not measured

        MPI_Send(sdata, ssize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPIDEVICESYNCHRONIZERESP, __cudampi__communicators[omp_get_thread_num()]);
      }

      if (status.MPI_TAG == __cudampi__CUDAMPICPUDEVICESYNCHRONIZEREQ) {

        int measurepower;
        cudaError_t error = cudaErrorUnknown;

        MPI_Recv(&measurepower, 1, MPI_INT, 0, __cudampi__CUDAMPICPUDEVICESYNCHRONIZEREQ, __cudampi__communicators[omp_get_thread_num()], &status);

        // perform power measurement and attach it to the response

        size_t ssize = sizeof(cudaError_t) + sizeof(float);
        unsigned char sdata[ssize];

        // Synchronize all threads
        #pragma omp taskwait

        if (measurepower) {
          error = getCpuEnergyUsed(&lastEnergyMeasured, (float *)(sdata + sizeof(cudaError_t)));
        }

        if (error != cudaSuccess) {
          *((float *)(sdata + sizeof(cudaError_t))) = -1;
        }

        *((cudaError_t *)sdata) = error;

        MPI_Send(sdata, ssize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPICPUDEVICESYNCHRONIZERESP, __cudampi__communicators[omp_get_thread_num()]);
      }

      if (status.MPI_TAG == __cudampi__CUDAMPISETDEVICEREQ) {

        int device;

        MPI_Recv(&device, 1, MPI_INT, 0, __cudampi__CUDAMPISETDEVICEREQ, __cudampi__communicators[omp_get_thread_num()], &status);

        cudaError_t e = cudaSetDevice(device);

        MPI_Send((unsigned char *)(&e), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPISETDEVICERESP, __cudampi__communicators[omp_get_thread_num()]);
      }

      if (status.MPI_TAG == __cudampi__CUDAMPIHOSTTODEVICEREQ) {

        // in this case in the message there is a serialized pointer and data so we need to find out the size first

        int rsize;
        MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &rsize);

        unsigned char rdata[rsize];

        MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPIHOSTTODEVICEREQ, __cudampi__communicators[omp_get_thread_num()], &status);

        void *devPtr = *((void **)rdata);

        // now send the data to the GPU
        cudaError_t e = cudaMemcpy(devPtr, rdata + sizeof(void *), rsize - sizeof(void *), cudaMemcpyHostToDevice);

        if (cudaSuccess != cudaGetLastError()) {
          printf("\nError xxx host to dev");
        }

        MPI_Send((unsigned char *)(&e), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPIHOSTTODEVICERESP, __cudampi__communicators[omp_get_thread_num()]);
      }

      if (status.MPI_TAG == __cudampi__CUDAMPIHOSTTODEVICEASYNCREQ) {

        // in this case in the message there is a serialized pointer and data so we need to find out the size first

        int rsize;
        MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &rsize);

        unsigned char rdata[rsize];

        MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPIHOSTTODEVICEASYNCREQ, __cudampi__communicators[omp_get_thread_num()], &status);

        void *devPtr = *((void **)rdata);
        cudaStream_t stream = *((cudaStream_t *)(rdata + sizeof(void *)));

        // now send the data to the GPU
        cudaError_t e = cudaMemcpyAsync(devPtr, rdata + sizeof(void *) + sizeof(cudaStream_t), rsize - sizeof(void *) - sizeof(cudaStream_t), cudaMemcpyHostToDevice, stream);

        if (cudaSuccess != cudaGetLastError()) {
          printf("\nError xxx host to dev async");
        }

        MPI_Send((unsigned char *)(&e), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPIHOSTTODEVICEASYNCRESP, __cudampi__communicators[omp_get_thread_num()]);
      }

      if (status.MPI_TAG == __cudampi__CUDAMPIDEVICETOHOSTREQ) {

        // in this case in the message there is a serialized pointer and size of data to fetch

        int rsize = sizeof(void *) + sizeof(unsigned long);
        unsigned char rdata[rsize];

        MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPIDEVICETOHOSTREQ, __cudampi__communicators[omp_get_thread_num()], &status);

        void *devPtr = *((void **)rdata);
        unsigned long count = *((unsigned long *)(rdata + sizeof(void *)));

        size_t ssize = sizeof(cudaError_t) + count;
        unsigned char sdata[ssize];

        // now send the data to the GPU
        cudaError_t e = cudaMemcpy(sdata + sizeof(cudaError_t), devPtr, count, cudaMemcpyDeviceToHost);

        if (cudaSuccess != cudaGetLastError()) {
          printf("\nError yyy dev to host");
        }

        *((cudaError_t *)sdata) = e;

        MPI_Send(sdata, ssize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPIDEVICETOHOSTRESP, __cudampi__communicators[omp_get_thread_num()]);
      }

      if (status.MPI_TAG == __cudampi__CUDAMPIDEVICETOHOSTASYNCREQ) {

        // in this case in the message there is a serialized pointer and size of data to fetch

        int rsize = sizeof(void *) + sizeof(unsigned long) + sizeof(cudaStream_t);
        unsigned char rdata[rsize];

        MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPIDEVICETOHOSTASYNCREQ, __cudampi__communicators[omp_get_thread_num()], &status);

        void *devPtr = *((void **)rdata);
        unsigned long count = *((unsigned long *)(rdata + sizeof(void *)));
        cudaStream_t stream = *((cudaStream_t *)(rdata + sizeof(void *) + sizeof(unsigned long)));

        size_t ssize = sizeof(cudaError_t) + count;
        unsigned char sdata[ssize];

        // now send the data to the GPU
        cudaError_t e = cudaMemcpyAsync(sdata + sizeof(cudaError_t), devPtr, count, cudaMemcpyDeviceToHost, stream);

        if (cudaSuccess != cudaGetLastError()) {
          printf("Error yyy dev to host async");
        }

        *((cudaError_t *)sdata) = e;

        MPI_Send(sdata, ssize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPIDEVICETOHOSTASYNCRESP, __cudampi__communicators[omp_get_thread_num()]);
      }

      if (status.MPI_TAG == __cudampi__CUDAMPILAUNCHCUDAKERNELREQ) {

        // in this case in the message there is a serialized pointer

        int rsize = sizeof(void *);
        unsigned char rdata[rsize];

        MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPILAUNCHCUDAKERNELREQ, __cudampi__communicators[omp_get_thread_num()], &status);

        void *devPtr = *((void **)rdata);

        launchkernel(devPtr);

        size_t ssize = sizeof(cudaError_t);
        unsigned char sdata[ssize];
        *((cudaError_t *)sdata) = cudaSuccess;

        MPI_Send(sdata, ssize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPILAUNCHCUDAKERNELRESP, __cudampi__communicators[omp_get_thread_num()]);
      }

      if (status.MPI_TAG == __cudampi__CUDAMPILAUNCHCPUKERNELREQ) {

        // in this case in the message there is a serialized pointer

        int rsize = sizeof(void *);
        unsigned char rdata[rsize];

        MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPILAUNCHCPUKERNELREQ, __cudampi__communicators[omp_get_thread_num()], &status);

        void *devPtr = *((void **)rdata);


        // TODO: Investigate if taskwait / lock mechanism should be implemented here
        #pragma omp task
        {
          // Launch with all free CPU threads - 1 for thread that manages CPU computation
          launchcpukernel(devPtr, __cudampi__localFreeThreadCount - 1);
        }

        MPI_Send(NULL, 0, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPILAUNCHCPUKERNELRESP, __cudampi__communicators[omp_get_thread_num()]);
      }

      if (status.MPI_TAG == __cudampi__CUDAMPILAUNCHKERNELINSTREAMREQ) {

        // in this case in the message there is a serialized pointer

        int rsize = sizeof(void *) + sizeof(cudaStream_t);
        unsigned char rdata[rsize];

        MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPILAUNCHKERNELINSTREAMREQ, __cudampi__communicators[omp_get_thread_num()], &status);

        void *devPtr = *((void **)rdata);
        cudaStream_t stream = *((cudaStream_t *)(rdata + sizeof(void *)));

        launchkernelinstream(devPtr, stream);

        size_t ssize = sizeof(cudaError_t);
        unsigned char sdata[ssize];
        *((cudaError_t *)sdata) = cudaSuccess;

        MPI_Send(sdata, ssize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPILAUNCHKERNELINSTREAMRESP, __cudampi__communicators[omp_get_thread_num()]);
      }

      if (status.MPI_TAG == __cudampi__CUDAMPISTREAMCREATEREQ) {

        MPI_Recv(NULL, 0, MPI_UNSIGNED_LONG, 0, __cudampi__CUDAMPISTREAMCREATEREQ, __cudampi__communicators[omp_get_thread_num()], &status);

        // create a new stream on the current GPU

        cudaStream_t pStream;

        cudaError_t e = cudaStreamCreate(&pStream);

        int ssize = sizeof(cudaStream_t) + sizeof(cudaError_t);
        // send confirmation with the actual pointer
        unsigned char sdata[ssize];

        *((cudaStream_t *)sdata) = pStream;
        *((cudaError_t *)(sdata + sizeof(cudaStream_t))) = e;

        MPI_Send(sdata, ssize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPISTREAMCREATERESP, __cudampi__communicators[omp_get_thread_num()]);
      }

      if (status.MPI_TAG == __cudampi__CUDAMPISTREAMDESTROYREQ) {

        int rsize = sizeof(cudaStream_t);
        unsigned char rdata[rsize];

        MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPISTREAMDESTROYREQ, __cudampi__communicators[omp_get_thread_num()], &status);

        cudaStream_t stream = *((cudaStream_t *)rdata);

        cudaError_t e = cudaStreamDestroy(stream);

        MPI_Send((unsigned char *)(&e), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPISTREAMDESTROYRESP, __cudampi__communicators[omp_get_thread_num()]);
      }

    } while (status.MPI_TAG != __cudampi__CUDAMPIFINALIZE);
  }

  MPI_Finalize();
}
