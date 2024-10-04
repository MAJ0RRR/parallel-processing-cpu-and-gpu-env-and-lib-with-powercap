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

#define ENABLE_LOGGING
#include "logger.h"

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

typedef struct dep_node {
    int dep_var;
    struct dep_node *next;
    struct dep_node *prev;
} dep_node_t;

dep_node_t *last_dep = NULL; // Pointer to the last dependency node
omp_lock_t dep_lock;

void launchkernel(void *devPtr);
void launchkernelinstream(void *devPtr, cudaStream_t stream);
void launchcpukernel(void *devPtr, int thread_count);

typedef struct cpu_malloc_args {
  unsigned long rdata;
  MPI_Comm* comm;
} cpu_malloc_args_t;

typedef struct cpu_free_args {
  void *devPtr;
  MPI_Comm* comm;
} cpu_free_args_t;

typedef struct cpu_host_to_device_args {
  void *devPtr;
  void *srcPtr;
  size_t count;
  MPI_Comm* comm;
} cpu_host_to_device_args_t;

typedef struct cpu_device_to_host_args {
  void *devPtr;
  unsigned long count;
  MPI_Comm* comm;
} cpu_device_to_host_args_t;

typedef struct cpu_launch_kernel_args {
  void *devPtr;
  MPI_Comm* comm;
} cpu_launch_kernel_args_t;

void cpuMallocTask(void* arg) {
  cpu_malloc_args_t *args = (cpu_malloc_args_t*) arg;

  void *devPtr = malloc((size_t)args->rdata);

  int ssize = sizeof(void *) + sizeof(cudaError_t);
  // send confirmation with the actual pointer
  unsigned char sdata[ssize];

  *((void **)sdata) = devPtr;

  // return cudaSuccess if memory is not NULL, cudaErrorMemoryAllocation otherwise
  cudaError_t e = (devPtr == NULL) ? cudaErrorMemoryAllocation : cudaSuccess;
  *((cudaError_t *)(sdata + sizeof(void *))) = e;

  MPI_Send(sdata, ssize, MPI_UNSIGNED_CHAR, 0, __cudampi__CPUMALLOCRESP, *(args->comm));
  free(arg);
}

void cpuFreeTask(void* arg) {
  cpu_free_args_t *args = (cpu_free_args_t*) arg;
  cudaError_t e = cudaErrorInvalidDevicePointer;

  if (args->devPtr != NULL) {
    free(args->devPtr);
    e = cudaSuccess;
  }

  MPI_Send((unsigned char *)(&e), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 0, __cudampi__CPUFREERESP, *(args->comm));

  free(arg);
}

void cpuHostToDeviceTask(void* arg) {
  cpu_host_to_device_args_t *args = (cpu_host_to_device_args_t*) arg;
  cudaError_t e = cudaErrorInvalidValue;

  if (args->devPtr != NULL && args->srcPtr != NULL && args->count > 0) {
    memcpy(args->devPtr, args->srcPtr, args->count);
    e = cudaSuccess;
  }

  MPI_Send((unsigned char *)(&e), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 0, __cudampi__CPUHOSTTODEVICERESP, *(args->comm));

  free(arg);
}

void cpuDeviceToHostTask(void* arg) {
  cpu_device_to_host_args_t *args = (cpu_device_to_host_args_t*) arg;
  cudaError_t e = cudaErrorInvalidValue;

  size_t ssize = sizeof(cudaError_t) + args->count;
  unsigned char sdata[ssize];

  if (args->devPtr != NULL && args->count > 0) {
    memcpy(sdata + sizeof(cudaError_t), args->devPtr, args->count);
    e = cudaSuccess;
  }

  *((cudaError_t *)sdata) = e;
  MPI_Send(sdata, ssize, MPI_UNSIGNED_CHAR, 0, __cudampi__CPUDEVICETOHOSTRESP, *(args->comm));

  free(arg);
}

void cpuHostToDeviceTaskAsync(void* arg) {
  cpu_host_to_device_args_t *args = (cpu_host_to_device_args_t*) arg;

  memcpy(args->devPtr, args->srcPtr, args->count);

  free(arg);
}

void cpuDeviceToHostTaskAsync(void* arg) {
  cpu_device_to_host_args_t *args = (cpu_device_to_host_args_t*) arg;

  size_t ssize = sizeof(cudaError_t) + args->count;
  unsigned char sdata[ssize];

  memcpy(sdata + sizeof(cudaError_t), args->devPtr, args->count);

  free(arg);
}
      
void cpuLaunchKernelTask(void* arg) {
  cpu_launch_kernel_args_t *args = (cpu_launch_kernel_args_t*) arg;

  launchcpukernel(args->devPtr, __cudampi__localFreeThreadCount - 1);

  MPI_Send(NULL, 0, MPI_UNSIGNED_CHAR, 0, __cudampi__CPULAUNCHKERNELRESP, *(args->comm));

  free(arg);
}

void allocateCpuTask(void (*task_func)(void *), void *arg, int id)
{
  // TODO: Maybe reference counting would be a simpler approach ?

  // The idea here is to keep a list of task dependencies
  // If new task executes and there are no previous dependencies,
  // just execute it and keep the list entry
  // When new task executes and there are previous dependencies,
  // execute it and after finishing clear previous entry
  //
  // EXAMPLE: 3 tasks are scheduled one after another
  // 
  // 1) First task is scheduled and dependency created
  // 2) Second task is scheduled and dependency added to list
  // 3) First task finishes and does nothing since there's next dependency
  // 4) Second task start sexecuting
  // 5) Third task is scheduled and dependency added to list
  // 6) Second Task finishes, removing first dependency
  // 7) Third task finishes, removing second dependency and itself since there are no next dependencies
  // 8) Dependency list is empty
  dep_node_t *new_dep = (dep_node_t *)malloc(sizeof(dep_node_t));
  if (new_dep == NULL) {
    fprintf(stderr, "Error allocating memory for dependency node.\n");
    return;
  }
  new_dep->dep_var = 0;
  new_dep->next = NULL;
  new_dep->prev = NULL;

  // Lock to safely update the dependency chain
  omp_set_lock(&dep_lock);
  new_dep->prev = last_dep;

  if (last_dep != NULL) {
    last_dep->next = new_dep;
  }

  last_dep = new_dep;
  omp_unset_lock(&dep_lock);
  
  
  printf("Created CPU task dependency node. Task ID = %d\n", id);
  fflush(stdout);
  if (new_dep->prev == NULL) {
      printf("DEP = NULL\n");
  fflush(stdout);
    // First task has no 'in' dependency
    #pragma omp task untied depend(out: new_dep->dep_var)
    {
      printf("Launching CPU task. Task ID = %d\n", id);
  fflush(stdout);
      // Execute the task function
      task_func(arg);

      // Free the current dependency node if there are no other tasks depending on it
      omp_set_lock(&dep_lock);
      if (new_dep->next == NULL)
      {
        free(new_dep);
        // there are no next dependencies, so list is empty
        last_dep = NULL;
      }
      omp_unset_lock(&dep_lock);
      printf("Finished CPU task execution. Task ID = %d\n", id);
  fflush(stdout);
    }
      printf("AFTER TASK\n");
  fflush(stdout);
  } else {
      printf("DEP != NULL\n");
  fflush(stdout);
      // Subsequent tasks depend on the previous task
      #pragma omp task untied depend(in: new_dep->prev->dep_var) depend(out: new_dep->dep_var)
      {
        printf("Launching CPU task. Task ID = %d\n", id);
  fflush(stdout);
        // Execute the task function
        task_func(arg);

        omp_set_lock(&dep_lock);

        // Free the previous dependency node if it hasn't been freed
        if (new_dep->prev != NULL) {
          if (new_dep->prev->next == new_dep) {
            free(new_dep->prev);
            new_dep->prev = NULL;
          }
        }

        // Free the current dependency node if there are no other tasks depending on it
        if (new_dep->next == NULL) {
          free(new_dep);
          // there are no next dependencies, so list is empty
          last_dep = NULL;
        }

        omp_unset_lock(&dep_lock);
        printf("Finished CPU task execution. Task ID = %d\n", id);
  fflush(stdout);
      }
      
      printf("AFTER TASK\n");
  fflush(stdout);
  }
}

int main(int argc, char **argv) {

  // basically this is a slave process that waits for requests and redirects
  // those to local GPU(s)

  omp_init_lock(&dep_lock);
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
  #pragma omp parallel num_threads(__cudampi__localGpuDeviceCount + numberOfCpuThreads + 1)
  {

if (omp_get_thread_num() < __cudampi__localGpuDeviceCount + numberOfCpuThreads)
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

      if (status.MPI_TAG == __cudampi__CPUMALLOCREQ) {

        unsigned long rdata;

        MPI_Recv((unsigned long *)(&rdata), 1, MPI_UNSIGNED_LONG, 0, __cudampi__CPUMALLOCREQ, __cudampi__communicators[omp_get_thread_num()], &status);

        cpu_malloc_args_t* args = malloc(sizeof(cpu_malloc_args_t));
        args->rdata = rdata;
        args->comm = &__cudampi__communicators[omp_get_thread_num()];
        printf("Allocating CPU task for __cudampi__CPUMALLOCREQ\n");
        fflush(stdout);
        allocateCpuTask(cpuMallocTask, args, 0);
      }

      if (status.MPI_TAG == __cudampi__CUDAMPIFREEREQ) {

        int rsize = sizeof(void *);
        unsigned char rdata[rsize];

        MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPIFREEREQ, __cudampi__communicators[omp_get_thread_num()], &status);

        void *devPtr = *((void **)rdata);

        cudaError_t e = cudaFree(devPtr);

        MPI_Send((unsigned char *)(&e), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPIFREERESP, __cudampi__communicators[omp_get_thread_num()]);
      }

      if (status.MPI_TAG == __cudampi__CPUFREEREQ) {
        int rsize = sizeof(void *);
        unsigned char rdata[rsize];

        MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CPUFREEREQ, __cudampi__communicators[omp_get_thread_num()], &status);

        cpu_free_args_t* args = malloc(sizeof(cpu_free_args_t));
        args->devPtr = *((void **)rdata);
        args->comm = &__cudampi__communicators[omp_get_thread_num()];

        printf("Allocating CPU task for __cudampi__CPUFREEREQ\n");
  fflush(stdout);
        allocateCpuTask(cpuFreeTask, args, 1);
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

      if (status.MPI_TAG == __cudampi__CPUHOSTTODEVICEREQ) {
        int rsize;
        MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &rsize);

        unsigned char rdata[rsize];
        MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CPUHOSTTODEVICEREQ, __cudampi__communicators[omp_get_thread_num()], &status);

        cpu_host_to_device_args_t* args = malloc(sizeof(cpu_host_to_device_args_t));
        args->devPtr = *((void **)rdata);
        args->srcPtr = rdata + sizeof(void *);
        args->count = rsize - sizeof(void *);
        args->comm = &__cudampi__communicators[omp_get_thread_num()];

        printf("Allocating CPU task for __cudampi__CPUHOSTTODEVICEREQ\n");
  fflush(stdout);
        allocateCpuTask(cpuHostToDeviceTask, args, 2);
      }

      if (status.MPI_TAG == __cudampi__CPUDEVICETOHOSTREQ) {
        int rsize = sizeof(void *) + sizeof(unsigned long);
        unsigned char rdata[rsize];

        MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CPUDEVICETOHOSTREQ, __cudampi__communicators[omp_get_thread_num()], &status);

        cpu_device_to_host_args_t* args = malloc(sizeof(cpu_device_to_host_args_t));
        args->devPtr = *((void **)rdata);
        args->count = *((unsigned long *)(rdata + sizeof(void *)));
        args->comm = &__cudampi__communicators[omp_get_thread_num()];

        printf("Allocating CPU task for __cudampi__CPUDEVICETOHOSTREQ\n");
  fflush(stdout);
        allocateCpuTask(cpuDeviceToHostTask, args, 3);
      }

      if (status.MPI_TAG == __cudampi__CPUHOSTTODEVICEREQASYNC) {
        cudaError_t e = cudaErrorInvalidValue;
        int rsize;
        MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &rsize);

        unsigned char rdata[rsize];
        MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CPUHOSTTODEVICEREQASYNC, __cudampi__communicators[omp_get_thread_num()], &status);

        cpu_host_to_device_args_t* args = malloc(sizeof(cpu_host_to_device_args_t));
        args->devPtr = *((void **)rdata);
        args->srcPtr = rdata + sizeof(void *);
        args->count = rsize - sizeof(void *);
        args->comm = &__cudampi__communicators[omp_get_thread_num()];

        if (args->devPtr != NULL && args->srcPtr != NULL && args->count > 0) {
          printf("Allocating CPU task for __cudampi__CPUHOSTTODEVICEREQASYNC\n");
  fflush(stdout);
          allocateCpuTask(cpuHostToDeviceTaskAsync, args, 4);
          e = cudaSuccess;
        }

        MPI_Send((unsigned char *)(&e), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 0, __cudampi__CPUHOSTTODEVICERESPASYNC, __cudampi__communicators[omp_get_thread_num()]);

      }

      if (status.MPI_TAG == __cudampi__CPUDEVICETOHOSTREQASYNC) {
        cudaError_t e = cudaErrorInvalidValue;
        int rsize = sizeof(void *) + sizeof(unsigned long);
        unsigned char rdata[rsize];

        MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CPUDEVICETOHOSTREQASYNC, __cudampi__communicators[omp_get_thread_num()], &status);

        cpu_device_to_host_args_t* args = malloc(sizeof(cpu_device_to_host_args_t));
        args->devPtr = *((void **)rdata);
        args->count = *((unsigned long *)(rdata + sizeof(void *)));
        args->comm = &__cudampi__communicators[omp_get_thread_num()];

        if (args->devPtr != NULL && args->count > 0) {
          printf("Allocating CPU task for __cudampi__CPUDEVICETOHOSTREQASYNC\n");
  fflush(stdout);
          allocateCpuTask(cpuDeviceToHostTaskAsync, args, 5);
          e = cudaSuccess;
        }

        MPI_Send((unsigned char *)(&e), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 0, __cudampi__CPUDEVICETOHOSTRESPASYNC, __cudampi__communicators[omp_get_thread_num()]);
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

      if (status.MPI_TAG == __cudampi__CPULAUNCHKERNELREQ) {
        int rsize = sizeof(void *);
        unsigned char rdata[rsize];

        MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CPULAUNCHKERNELREQ, __cudampi__communicators[omp_get_thread_num()], &status);

        cpu_launch_kernel_args_t* args = malloc(sizeof(cpu_launch_kernel_args_t));
        args->devPtr = *((void **)rdata);
        args->comm = &__cudampi__communicators[omp_get_thread_num()];

        printf("Allocating CPU task for __cudampi__CPULAUNCHKERNELREQ\n");
  fflush(stdout);
        allocateCpuTask(cpuLaunchKernelTask, args, 6);
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
else
{
  while(1)
  {
     // TODO add handling CPU tasks here
  }
}

  MPI_Finalize();
  omp_destroy_lock(&dep_lock);
}
}
