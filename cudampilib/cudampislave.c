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
#include <sys/queue.h>

#include "cudampi.h"
#include "cudampicommon.h"

#define CPU_STREAMS_SUPPORTED 1
#define ENABLE_LOGGING
#include "logger.h"

int __cudampi__MPIproccount;
int __cudampi__myrank;
int terminated = 0;

int debugTaskCounter = 0;

float lastEnergyMeasured = 0.0;

int *__cudampi_targetMPIrankfordevice; // MPI rank for device number (global)
int *__cudampi__GPUcountspernode;
int *__cudampi__freeThreadsPerNode;

MPI_Comm *__cudampi__communicators;

int __cudampi_totaldevicecount = 0; // how many GPUs in total (on all considered nodes)
int __cudampi__localGpuDeviceCount = 1;
int __cudampi__localFreeThreadCount = 0;

unsigned long cpuStreamsValid[CPU_STREAMS_SUPPORTED];

typedef struct task_queue_entry {
    TAILQ_ENTRY(task_queue_entry) entries; // Use sys/queue.h macros for queue entries
    void (*task_func)(void *);
    void *arg;
    int id;
    int dep_var;   // Dependency variable for OpenMP
    int scheduled;    // Flag to indicate if the task has been scheduled
} task_queue_entry_t;

TAILQ_HEAD(task_queue_head, task_queue_entry) task_queue;
omp_lock_t queue_lock;

// CPU task launcher thread will wait on this lock until there are tasks available
// When there are no tasks in the queue, the lock is set
// When new task is being added to the queue, lock is being unset
omp_lock_t task_available_lock;
omp_lock_t synchronize_lock;

int scheduledTasksInQueue = 0;
void launchkernel(void *devPtr);
void launchkernelinstream(void *devPtr, cudaStream_t stream);
void launchcpukernel(void *devPtr, int thread_count);

typedef struct cpu_host_to_device_args {
  void *devPtr;
  size_t count;
  MPI_Comm* comm;
} cpu_host_to_device_args_t;

typedef struct cpu_device_to_host_args {
  void *devPtr;
  unsigned long count;
  MPI_Comm* comm;
} cpu_device_to_host_args_t;

void cpuSynchronize()
{
  log_message(LOG_DEBUG, "Synchronizing CPU tasks");
  // Check if there are tasks waiting to be synchronized
  omp_set_lock(&synchronize_lock);
  // Free the lock back
  omp_unset_lock(&synchronize_lock);
}

void cpuHostToDeviceTaskAsync(void* arg) {
  cudaError_t e = cudaErrorInvalidValue;
  cpu_host_to_device_args_t *args = (cpu_host_to_device_args_t*) arg;

  if (args->devPtr != NULL && args->count > 0){
    e = cudaSuccess;
    MPI_Recv((unsigned char *)args->devPtr, args->count, MPI_UNSIGNED_CHAR, 0, __cudampi__HOSTTODEVICEDATA, *(args->comm), NULL);
  }

  MPI_Send((unsigned char *)(&e), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 0, __cudampi__HOSTTODEVICEASYNCRESP, *(args->comm));

  free(arg);
}

void cpuDeviceToHostTaskAsync(void* arg) {
  cudaError_t e = cudaErrorInvalidValue;
  cpu_device_to_host_args_t *args = (cpu_device_to_host_args_t*) arg;
  
  if (args->devPtr != NULL && args->count > 0){
    e = cudaSuccess;
    MPI_Send((unsigned char*)(args->devPtr), args->count , MPI_UNSIGNED_CHAR, 0, __cudampi__DEVICETOHOSTDATA, *(args->comm));
  }

  MPI_Send((unsigned char *)(&e), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 0, __cudampi__DEVICETOHOSTASYNCRESP, *(args->comm));

  free(arg);
}

void cpuLaunchKernelTask(void* arg) {
  // kernel just takes void*
  launchcpukernel(arg, __cudampi__localFreeThreadCount - 1);
}

void allocateCpuTaskInStream(void (*task_func)(void *), void *arg, unsigned long stream)
{
  // This function takes a function pointer and argument and adds task to execute it to the list
  // Execution can be done remotely by another thread signaled with task_available_lock

  // Try setting lock if it is not set (no tasks in the queue);
  omp_test_lock(&synchronize_lock);

  task_queue_entry_t *new_dep = (task_queue_entry_t *)malloc(sizeof(task_queue_entry_t));
  if (new_dep == NULL) {
    log_message(LOG_ERROR, "Error allocating memory for dependency node.\n");
    return;
  }
  new_dep->dep_var = 0;  // Initialize dep_var
  new_dep->id = debugTaskCounter++; // Assign ID and increment the counter
  new_dep->task_func = task_func;
  new_dep->arg = arg;
  new_dep->scheduled = 0;   // Task is not scheduled yet

  // Lock to safely update the dependency chain
  omp_set_lock(&queue_lock);

  TAILQ_INSERT_TAIL(&task_queue, new_dep, entries);

  // Signal the task_launcher that a task is available
  // Unlock task_available_lock to allow task_launcher to proceed
  log_message(LOG_DEBUG, "Unsetting task lock\n");
  omp_unset_lock(&task_available_lock);

  omp_unset_lock(&queue_lock);

  log_message(LOG_DEBUG, "Created CPU task dependency node. Task ID = %d\n", new_dep->id);
}

void scheduleCpuTask(task_queue_entry_t* current_task_queue_entry)
{
  // This function exists, so that independent copy of current_task_queue_entry would be created.
  // Caller of this function moves head pointer of the list by one, but does not remove the node,
  // so there can be a situation that list will be empty (head == tail == NULL),
  // but there will be multiple existing nodes waiting for execution.
  // Nodes can be freed only after the next node after it finished execution,
  // so actual memory management (freeing not used nodes, setting pointer to NULL) is done
  // inside tasks started in this function, meaning each task frees it's previous node.

  task_queue_entry_t* prev_node;

  // IMPORTANT: The assumption here is that queue lock is already set when this is called 
  prev_node = TAILQ_PREV(current_task_queue_entry, task_queue_head, entries);
  current_task_queue_entry->scheduled = 1;
  scheduledTasksInQueue += 1;

  if (prev_node == NULL)
  {
    // First task has no 'in' dependency
    #pragma omp task untied depend(out: current_task_queue_entry->dep_var)
    {
      log_message(LOG_DEBUG, "FIRST Launching CPU task. Task ID = %d\n", current_task_queue_entry->id);

      // Execute the task function
      current_task_queue_entry->task_func(current_task_queue_entry->arg);

      omp_set_lock(&queue_lock);

      // Free the current dependency node if there are no other tasks depending on it
      if (TAILQ_NEXT(current_task_queue_entry, entries) == NULL){
        TAILQ_REMOVE(&task_queue, current_task_queue_entry, entries);
        
        // No next task meaning queue is empty
        // Set task_available lock here, so that there wouldn't be a scenario
        // that between now and new lock set there would be a task added
        log_message(LOG_DEBUG, "Unsetting task lock inisde launcher\n");
        
        omp_unset_lock(&synchronize_lock);
        omp_test_lock(&task_available_lock); 
      }
      scheduledTasksInQueue -= 1;
      omp_unset_lock(&queue_lock);

      log_message(LOG_DEBUG, "Finished CPU task execution. Task ID = %d\n", current_task_queue_entry->id);
    }
  } else {
    // Subsequent tasks depend on the previous task
    #pragma omp task untied depend(in: prev_node->dep_var) depend(out: current_task_queue_entry->dep_var)
    {
      log_message(LOG_DEBUG, "SECOND Launching CPU task. Task ID = %d\n", current_task_queue_entry->id);

      // Execute the task function
      current_task_queue_entry->task_func(current_task_queue_entry->arg);

      omp_set_lock(&queue_lock);

      //Free the previous dependency node if it hasn't been freed
      TAILQ_REMOVE(&task_queue, prev_node, entries);

      // Free the current dependency node if there are no other tasks depending on it
      if (TAILQ_NEXT(current_task_queue_entry, entries) == NULL){
        TAILQ_REMOVE(&task_queue, current_task_queue_entry, entries);

        // No next task meaning queue is empty
        // Set task_available lock here, so that there wouldn't be a scenario
        // that between now and new lock set there would be a task added
        log_message(LOG_DEBUG, "Unsetting task lock inisde launcher\n");
        
        omp_unset_lock(&synchronize_lock);
        omp_test_lock(&task_available_lock); 
      }

      scheduledTasksInQueue -= 1;
      omp_unset_lock(&queue_lock);

      log_message(LOG_DEBUG, "Finished CPU task execution. Task ID = %d\n", current_task_queue_entry->id);
    }
  }
}

void cpuTaskLauncher()
{
  // The assumption here is that there is only one thread executing this code.
  // This loop processes the entire FIFO by scheduling taks and exits.

  task_queue_entry_t *current_task_queue_entry;

  omp_set_lock(&queue_lock);

  // Iterate over the queue and schedule unscheduled tasks
  TAILQ_FOREACH(current_task_queue_entry, &task_queue, entries) {
      if (current_task_queue_entry->scheduled == 0) {
          scheduleCpuTask(current_task_queue_entry);
      }
  }
  omp_unset_lock(&queue_lock);
}

int main(int argc, char **argv) {

  // basically this is a slave process that waits for requests and redirects
  // those to local GPU(s)

  omp_init_lock(&queue_lock);
  omp_init_lock(&synchronize_lock);
  omp_init_lock(&task_available_lock);
  // Set the lock since there are no tasks in queue
  omp_set_lock(&task_available_lock);

  // Initialize the queue
  TAILQ_INIT(&task_queue);

  int mtsprovided;

  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mtsprovided);

  if (mtsprovided != MPI_THREAD_MULTIPLE) {
    log_message(LOG_ERROR, "\nNo support for MPI_THREAD_MULTIPLE mode.\n");
    exit(-1);
  }

  // enable nested omp parallels
  omp_set_nested(1);

  // fetch information about the rank and number of processes

  MPI_Comm_size(MPI_COMM_WORLD, &__cudampi__MPIproccount);
  MPI_Comm_rank(MPI_COMM_WORLD, &__cudampi__myrank);

  // now check the number of GPUs available and synchronize with process 0 that wants this info

  if (cudaSuccess != cudaGetDeviceCount(&__cudampi__localGpuDeviceCount)) {
    log_message(LOG_ERROR, "Error invoking cudaGetDeviceCount()");
    exit(-1);
  }

  __cudampi__GPUcountspernode = (int *)malloc(sizeof(int) * __cudampi__MPIproccount);
  if (!__cudampi__GPUcountspernode) {
    log_message(LOG_ERROR, "\nNot enough memory");
    exit(-1); // we could exit in a nicer way! TBD
  }

  __cudampi__freeThreadsPerNode = (int *)malloc(sizeof(int) * __cudampi__MPIproccount);
  if (!__cudampi__freeThreadsPerNode) {
    log_message(LOG_ERROR, "\nNot enough memory");
    exit(-1); // we could exit in a nicer way! TBD
  }

  if (cudaSuccess != __cudampi__getCpuFreeThreads(&__cudampi__localFreeThreadCount)) {
    log_message(LOG_ERROR, "Error invoking __cudampi__getCpuFreeThreads()");
    exit(-1);
  }

  MPI_Allgather(&__cudampi__localGpuDeviceCount, 1, MPI_INT, __cudampi__GPUcountspernode, 1, MPI_INT, MPI_COMM_WORLD);

  MPI_Allgather(&__cudampi__localFreeThreadCount, 1, MPI_INT, __cudampi__freeThreadsPerNode, 1, MPI_INT, MPI_COMM_WORLD);

  MPI_Bcast(&__cudampi_totaldevicecount, 1, MPI_INT, 0, MPI_COMM_WORLD);

  __cudampi_targetMPIrankfordevice = (int *)malloc(__cudampi_totaldevicecount * sizeof(int));
  if (!__cudampi_targetMPIrankfordevice) {
    log_message(LOG_ERROR, "\nNot enough memory");
    exit(-1); // we could exit in a nicer way! TBD
  }

  MPI_Bcast(__cudampi_targetMPIrankfordevice, __cudampi_totaldevicecount, MPI_INT, 0, MPI_COMM_WORLD);

  // create communicators
  // in the case of the slave we need to go by every GPU and for each GPU there will be a separate GPU shared with the master -- process 0

  MPI_Comm *__cudampi__communicators = (MPI_Comm *)malloc(sizeof(MPI_Comm) * __cudampi_totaldevicecount);
  if (!__cudampi__communicators) {
    log_message(LOG_ERROR, "\nNot enough memory for communicators");
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

  // here we need to spawn threads -- each responsible for handling one local GPU / CPU
  // spawn one thread for CPU processing if there are free cores, 0 otherwise
  int numberOfCpuThreads = __cudampi__localFreeThreadCount > 0;
  int deviceThreads = numberOfCpuThreads + __cudampi__localGpuDeviceCount; 
  int numberOfThreads = deviceThreads;
  
  // Add one thread for executing CPU operations asynchronously
  numberOfThreads += 1;

  #pragma omp parallel num_threads(numberOfThreads)
  {

    if (omp_get_thread_num() < deviceThreads)
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

        cpuSynchronize();
        log_message(LOG_DEBUG, "Executing synchronously CPU task for __cudampi__CPUMALLOCREQ\n");

        void *devPtr = malloc((size_t)rdata);

        int ssize = sizeof(void *) + sizeof(cudaError_t);
        // send confirmation with the actual pointer
        unsigned char sdata[ssize];

        *((void **)sdata) = devPtr;

        // return cudaSuccess if memory is not NULL, cudaErrorMemoryAllocation otherwise
        cudaError_t e = (devPtr == NULL) ? cudaErrorMemoryAllocation : cudaSuccess;
        *((cudaError_t *)(sdata + sizeof(void *))) = e;

        MPI_Send(sdata, ssize, MPI_UNSIGNED_CHAR, 0, __cudampi__CPUMALLOCRESP, __cudampi__communicators[omp_get_thread_num()]);
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

        cpuSynchronize();
        log_message(LOG_DEBUG, "Executing synchronously CPU task for __cudampi__CPUFREEREQ\n");

        void *devPtr = *((void **)rdata);
        cudaError_t e = cudaErrorInvalidDevicePointer;

        // maybe we should somehow handle invalid devPtr ?
        if (devPtr != NULL) {
          free(devPtr);
          e = cudaSuccess;
        }

        MPI_Send((unsigned char *)(&e), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 0, __cudampi__CPUFREERESP, __cudampi__communicators[omp_get_thread_num()]);
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

        cpuSynchronize();

        if (measurepower) {
          error = getCpuEnergyUsed(&lastEnergyMeasured, (float *)(sdata + sizeof(cudaError_t)));
        }

        if (error != cudaSuccess) {
          *((float *)(sdata + sizeof(cudaError_t))) = -1;
        }

        *((cudaError_t *)sdata) = error;

        
        log_message(LOG_DEBUG, "Synchronized CPU device\n");
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
          log_message(LOG_ERROR, "\nError xxx host to dev");
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
          log_message(LOG_ERROR, "\nError xxx host to dev async");
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
          log_message(LOG_ERROR, "\nError yyy dev to host");
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
          log_message(LOG_ERROR, "Error yyy dev to host async");
        }

        *((cudaError_t *)sdata) = e;

        MPI_Send(sdata, ssize, MPI_UNSIGNED_CHAR, 0, __cudampi__CUDAMPIDEVICETOHOSTASYNCRESP, __cudampi__communicators[omp_get_thread_num()]);
      }

      if (status.MPI_TAG == __cudampi__CPUHOSTTODEVICEREQ) {
        int rsize;
        MPI_Get_count(&status, MPI_UNSIGNED_CHAR, &rsize);

        unsigned char rdata[rsize];
        MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CPUHOSTTODEVICEREQ, __cudampi__communicators[omp_get_thread_num()], &status);

        cpuSynchronize();
        log_message(LOG_DEBUG, "Executing synchronously CPU task for __cudampi__CPUHOSTTODEVICEREQ\n");

        void *devPtr = *((void **)rdata);
        void *srcPtr = rdata + sizeof(void *);
        size_t count = rsize - sizeof(void *);

        cudaError_t e = cudaErrorInvalidValue;

        if (devPtr != NULL && srcPtr != NULL && count > 0)
        {
          memcpy(devPtr, srcPtr, rsize - sizeof(void *));
          e = cudaSuccess; 
        }

        MPI_Send((unsigned char *)(&e), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 0, __cudampi__CPUHOSTTODEVICERESP, __cudampi__communicators[omp_get_thread_num()]);
      }

      if (status.MPI_TAG == __cudampi__CPUDEVICETOHOSTREQ) {
        int rsize = sizeof(void *) + sizeof(unsigned long);
        unsigned char* rdata[rsize];

        MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CPUDEVICETOHOSTREQ, __cudampi__communicators[omp_get_thread_num()], &status);

        cpuSynchronize();
        log_message(LOG_DEBUG, "Executing synchronously CPU task for __cudampi__CPUDEVICETOHOSTREQ\n");

        void *devPtr = *((void **)rdata);
        unsigned long count = *((unsigned long *)(rdata + sizeof(void *)));

        cudaError_t e = cudaErrorInvalidValue;

        size_t ssize = sizeof(cudaError_t) + count;
        unsigned char sdata[ssize];

        if (devPtr != NULL && count > 0)
        {
          memcpy(sdata + sizeof(cudaError_t), devPtr, count);
          e = cudaSuccess; 
        }

        *((cudaError_t *)sdata) = e;

        MPI_Send(sdata, ssize, MPI_UNSIGNED_CHAR, 0, __cudampi__CPUDEVICETOHOSTRESP, __cudampi__communicators[omp_get_thread_num()]);
      }

      if (status.MPI_TAG == __cudampi__CPUHOSTTODEVICEREQASYNC) {
        int rsize = sizeof(void*) + sizeof(unsigned long) + sizeof(unsigned long);
        // Receive a request with number of bytes that will be sent and a pointer
        unsigned char rdata[rsize];
        MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CPUHOSTTODEVICEREQASYNC, __cudampi__communicators[omp_get_thread_num()], &status);

        // Launch a task that will receive the data
        cpu_host_to_device_args_t* args = malloc(sizeof(cpu_host_to_device_args_t));
        args->devPtr = *((void **)rdata);
        args->count = *((unsigned long*)(rdata + sizeof(void*)));
        args->comm = &__cudampi__communicators[omp_get_thread_num()];

        unsigned long stream = *((unsigned long*)(rdata + sizeof(void*) + sizeof(unsigned long)));
        log_message(LOG_DEBUG, "Allocating CPU task for __cudampi__CPUHOSTTODEVICEREQASYNC\n");
        allocateCpuTaskInStream(cpuHostToDeviceTaskAsync, args, stream);
      }

      if (status.MPI_TAG == __cudampi__CPUDEVICETOHOSTREQASYNC) {
        int rsize = sizeof(void *) + sizeof(unsigned long) + sizeof(unsigned long);
        unsigned char rdata[rsize];

        MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CPUDEVICETOHOSTREQASYNC, __cudampi__communicators[omp_get_thread_num()], &status);

        cpu_device_to_host_args_t* args = malloc(sizeof(cpu_device_to_host_args_t));
        args->devPtr = *((void **)rdata);
        args->count = *((unsigned long *)(rdata + sizeof(void *)));
        args->comm = &__cudampi__communicators[omp_get_thread_num()];

        unsigned long stream = *((unsigned long*)(rdata + sizeof(void*) + sizeof(unsigned long)));
        log_message(LOG_DEBUG, "Allocating CPU task for __cudampi__CPUDEVICETOHOSTREQASYNC\n");
        allocateCpuTaskInStream(cpuDeviceToHostTaskAsync, args, stream);
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
        int rsize = sizeof(void *) + sizeof(unsigned long);
        unsigned char rdata[rsize];

        MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CPULAUNCHKERNELREQ, __cudampi__communicators[omp_get_thread_num()], &status);

        unsigned long stream = *((unsigned long*)(rdata + sizeof(void*) ));
        log_message(LOG_DEBUG, "Allocating CPU task for __cudampi__CPULAUNCHKERNELREQ\n");
        allocateCpuTaskInStream(cpuLaunchKernelTask, *((void **)rdata), stream);
        
        MPI_Send(NULL, 0, MPI_UNSIGNED_CHAR, 0, __cudampi__CPULAUNCHKERNELRESP, __cudampi__communicators[omp_get_thread_num()]);
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

      if (status.MPI_TAG == __cudampi__CPUSTREAMCREATEREQ) {

        MPI_Recv(NULL, 0, MPI_UNSIGNED_LONG, 0, __cudampi__CPUSTREAMCREATEREQ, __cudampi__communicators[omp_get_thread_num()], &status);

        // create a new stream on CPU
        cudaError_t e = cudaErrorInvalidResourceHandle;

        int freeStreamSlot = -1;
        for (int i = 0; i < CPU_STREAMS_SUPPORTED; i++) {
          // Find free stream slot
          if (cpuStreamsValid[i] == 0)
          {
            cpuStreamsValid[i] = 1;
            freeStreamSlot = i;
            e = cudaSuccess;
            break;
          }
        }

        int ssize = sizeof(unsigned long) + sizeof(cudaError_t);
        // send confirmation with the actual pointer
        unsigned char sdata[ssize];

        *((unsigned long *)sdata) = freeStreamSlot;
        *((cudaError_t *)(sdata + sizeof(cudaStream_t))) = e;

        MPI_Send(sdata, ssize, MPI_UNSIGNED_CHAR, 0, __cudampi__CPUSTREAMCREATERESP, __cudampi__communicators[omp_get_thread_num()]);
      }

      if (status.MPI_TAG == __cudampi__CPUSTREAMDESTROYREQ) {
        int rsize = sizeof(unsigned long);
        unsigned char rdata[rsize];

        MPI_Recv((unsigned char *)rdata, rsize, MPI_UNSIGNED_CHAR, 0, __cudampi__CPUSTREAMDESTROYREQ, __cudampi__communicators[omp_get_thread_num()], &status);

        unsigned long stream = *((unsigned long *)rdata);

        cudaError_t e = cudaErrorInvalidResourceHandle;

        if(stream < CPU_STREAMS_SUPPORTED)
        {
          cpuStreamsValid[stream] = 0;
          e = cudaSuccess;
        }

        MPI_Send((unsigned char *)(&e), sizeof(cudaError_t), MPI_UNSIGNED_CHAR, 0, __cudampi__CPUSTREAMDESTROYRESP, __cudampi__communicators[omp_get_thread_num()]);
      }

    } while (status.MPI_TAG != __cudampi__CUDAMPIFINALIZE);
    terminated = 1;
    omp_unset_lock(&task_available_lock);
  }
else
{
  int localTaskCounter = 0 ;
  while(!terminated)
  {
    // If there's data in task queue, wait for completion
    omp_set_lock(&queue_lock);
    localTaskCounter = scheduledTasksInQueue;
    omp_unset_lock(&queue_lock);

    log_message(LOG_DEBUG,"Tasks in queue: %d\n", localTaskCounter);

    if (localTaskCounter > 0)
    {
      log_message(LOG_DEBUG, "Waiting for task execution \n");
      #pragma omp taskwait
    }
    else
    {
      
      log_message(LOG_DEBUG, "Setting task lock \n");
      // Wait for new tasks to be allocated
      omp_set_lock(&task_available_lock);
      // Call a function that schedules all the tasks in the queue
      cpuTaskLauncher();
    }
  }
  log_message(LOG_DEBUG, "Terminated task handling thread!");
  omp_destroy_lock(&task_available_lock);
}
}
  MPI_Finalize();
  omp_destroy_lock(&queue_lock);
}
