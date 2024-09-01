/*
Copyright 2023 Paweł Czarnul pczarnul@eti.pg.edu.pl

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <omp.h>

#include "cudampicommon.h"
#include "cudampilib.h"

// int __cudampi__GPUcountpernode=1;

#define __cudampi__currentDevice  __cudampi__currentdevice[omp_get_thread_num()]
#define __cudampi__currentCommunicator  __cudampi__communicators[__cudampi__currentDevice]

#define  __cudampi__isLocalCpu omp_get_thread_num() == __cudampi_totalgpudevicecount
#define  __cudampi_isLocalGpu __cudampi__currentDevice < __cudampi__GPUcountspernode[0]


int *__cudampi__GPUcountspernode;
int *__cudampi__freeThreadsPerNode;
int __cudampi_totaldevicecount = 0; // how many GPUs + CPUs (on all considered nodes)
int __cudampi_totalgpudevicecount = 0; // how many GPUs in total (on all considered nodes)
int __cudampi_totalcpudevicecount = 0; // how many CPUs in total (on all considered nodes)

int __cudampi_localdevicecount = 0; // how many GPUs in process 0

int *__cudampi_targetGPUfordevice;     // GPU id on a given node for device number (global)
int *__cudampi_targetMPIrankfordevice; // MPI rank for device number (global)

int __cudampi__MPIinitialized = 0;
int __cudampi__MPIproccount;
int __cudampi__myrank;

int __cudampi__currentdevice[__CUDAMPI_MAX_THREAD_COUNT]; // current device id for various threads in process 0

struct timeval __cudampi__timestart[__CUDAMPI_MAX_THREAD_COUNT]; // start of time measurement
struct timeval __cudampi__timestop[__CUDAMPI_MAX_THREAD_COUNT];  // end of time measurement
struct timeval __cudampi__time[__CUDAMPI_MAX_THREAD_COUNT];      // last time measurement (from start to stop)
int __cudampi__timemeasured[__CUDAMPI_MAX_THREAD_COUNT] = {0};   // whether time measurement started
float __cudampi__devicepower[__CUDAMPI_MAX_THREAD_COUNT];        // current power taken by a device

int __cudampi__deviceenabled[__CUDAMPI_MAX_THREAD_COUNT]; // whether the given device is enabled for further use

omp_lock_t __cudampi__devicelocks[__CUDAMPI_MAX_THREAD_COUNT]; // locks that guard writing to and reading from power and time values for particular devices

omp_lock_t deviceselectionlock;

int __cudampi__amimanager[__CUDAMPI_MAX_THREAD_COUNT] = {0}; // whether the given thread/device is the manager for device selection

MPI_Comm *__cudampi__communicators; // communicators for communication with threads responsible for target GPUs, there is one communicator for such target GPU

int __cudampi__isglobalpowerlimitset = 0; // whether a global power limit has been set
float __cudampi__globalpowerlimit;

int powermeasurecounter[__CUDAMPI_MAX_THREAD_COUNT] = {0};

void __cudampi__setglobalpowerlimit(float powerlimit) {

  __cudampi__isglobalpowerlimitset = 1;
  __cudampi__globalpowerlimit = powerlimit;
}

float __cudampi__gettotalpowerofselecteddevices() { // gets total power of currently enabled devices
  int i;
  float power = 0;
  float curpower;

  omp_set_lock(&deviceselectionlock);

  for (i = 0; i < __cudampi_totalgpudevicecount; i++) {
    omp_set_lock(&(__cudampi__devicelocks[__cudampi__currentdevice[i]]));
  }

  for (i = 0; i < __cudampi_totalgpudevicecount; i++) {
    if (__cudampi__deviceenabled[__cudampi__currentdevice[i]] == 1) {
      curpower = __cudampi__devicepower[__cudampi__currentdevice[i]];
      if (curpower == (-1)) {

        for (int i = 0; i < __cudampi_totalgpudevicecount; i++) {
          omp_unset_lock(&(__cudampi__devicelocks[__cudampi__currentdevice[i]]));
        }

        omp_unset_lock(&deviceselectionlock);
        return -1;
      }
      power += curpower;
    }
  }

  for (i = 0; i < __cudampi_totalgpudevicecount; i++) {
    omp_unset_lock(&(__cudampi__devicelocks[__cudampi__currentdevice[i]]));
  }

  omp_unset_lock(&deviceselectionlock);
  return power;
}

int __cudampi__selectdevicesforpowerlimit_greedy() { // adopts a greedy strategy for selecting devices
                                                     // returns 1 if successful, 0 otherwise - if not all devices have been recorder power
  int i;
  float powerleft;
  int indexselected;
  float curperfpower, temp;
  int anydeviceenabled = 0;

  printf("\nbefore");
  fflush(stdout);

  omp_set_lock(&deviceselectionlock);

  if (__cudampi__isglobalpowerlimitset == 0) {
    printf("\n no limite set");
    fflush(stdout);
    omp_unset_lock(&deviceselectionlock);
    return 0;
  }

  powerleft = __cudampi__globalpowerlimit;
  // this will be invoked from one thread typically
  printf("\naaa");
  fflush(stdout);
  for (i = 0; i < __cudampi_totalgpudevicecount; i++) {
    printf("\nsetting lock on %d %d", i, __cudampi__currentdevice[i]);
    fflush(stdout);
    omp_set_lock(&(__cudampi__devicelocks[__cudampi__currentdevice[i]]));
    // disable all devices at first
  }
  printf("\nbbb");
  fflush(stdout);

  // check of all the devices has been set power
  int allpowerset = 1;
  for (i = 0; i < __cudampi_totalgpudevicecount; i++) {
    if (__cudampi__devicepower[__cudampi__currentdevice[i]] == (-1)) {
      allpowerset = 0;
      break;
    }
  }

  if (!allpowerset) {
    // unlock and quit
    printf("before unlocking");
    fflush(stdout);
    for (i = 0; i < __cudampi_totalgpudevicecount; i++) {
      omp_unset_lock(&(__cudampi__devicelocks[__cudampi__currentdevice[i]]));
    }
    printf("not all set");
    fflush(stdout);

    omp_unset_lock(&deviceselectionlock);

    return 0;
  }

  for (i = 0; i < __cudampi_totalgpudevicecount; i++) {
    if (__cudampi__deviceenabled[__cudampi__currentdevice[i]] == 1) {
      __cudampi__deviceenabled[__cudampi__currentdevice[i]] = -1; // candidate for selection
    }
    __cudampi__amimanager[__cudampi__currentdevice[i]] = 0;
  }

  printf("\nggg");
  fflush(stdout);

  int managerselected = 0;
  do {
    curperfpower = 0;
    indexselected = -1;
    for (i = 0; i < __cudampi_totalgpudevicecount; i++) {
      if (((-1) == (__cudampi__deviceenabled[__cudampi__currentdevice[i]])) && (__cudampi__devicepower[__cudampi__currentdevice[i]] <= powerleft) &&
          ((temp = (computeDevPerformance(__cudampi__time[__cudampi__currentdevice[i]]) / __cudampi__devicepower[__cudampi__currentdevice[i]])) > curperfpower)) {
        curperfpower = temp;
        indexselected = i;
        anydeviceenabled = 1;
      }
    }
    if (indexselected != (-1)) {
      // enable the found device now
      __cudampi__deviceenabled[__cudampi__currentdevice[indexselected]] = 1;
      if (!managerselected) {
        managerselected = 1;
        __cudampi__amimanager[__cudampi__currentdevice[indexselected]] = 1;
      }
      powerleft -= __cudampi__devicepower[__cudampi__currentdevice[indexselected]];
      printf("\nSelected device %d", __cudampi__currentdevice[indexselected]);
    }
  } while (indexselected != (-1));
  printf("hhh");
  fflush(stdout);

  if (!anydeviceenabled) { // handle this case
    printf("No devices found under the power limit");
    fflush(stdout);
    exit(-1);
  }

  // now not enabled devices are set to 0
  for (i = 0; i < __cudampi_totalgpudevicecount; i++) {
    if (__cudampi__deviceenabled[__cudampi__currentdevice[i]] != 1) {
      __cudampi__deviceenabled[__cudampi__currentdevice[i]] = 0;
    }
  }

  // unlock the devices' locks
  for (i = 0; i < __cudampi_totalgpudevicecount; i++) {
    omp_unset_lock(&(__cudampi__devicelocks[__cudampi__currentdevice[i]]));
  }

  printf("\nafter");
  fflush(stdout);

  omp_unset_lock(&deviceselectionlock);

  return 1;
}

int __cudampi__getnextchunkindex(long long *globalcounter, int batchsize, long long max) { return __cudampi__getnextchunkindex_enableddevices(globalcounter, batchsize, max); }

int __cudampi__getnextchunkindex_enableddevices(long long *globalcounter, int batchsize, long long max) {
  // for a given thread (GPU) return the next available data chunk
  // max is the vector size
  long long mycounter;
  int deviceenabled;

  omp_set_lock(&(__cudampi__devicelocks[__cudampi__currentDevice]));
  deviceenabled = __cudampi__deviceenabled[__cudampi__currentDevice];
  omp_unset_lock(&(__cudampi__devicelocks[__cudampi__currentDevice]));

  if (deviceenabled == 1) {

    #pragma omp critical
    {
      mycounter = *globalcounter; // handle data from mycounter to mycounter+batchsize-1
      (*globalcounter) += batchsize;
    }
  } else {
    mycounter = max; // force not giving any more data to the device
  }

  return mycounter;
}

int __cudampi__getnextchunkindex_alldevices(long long *globalcounter, int batchsize, long long max) {
  // for a given thread (GPU) return the next available data chunk
  // max is the vector size
  long long mycounter;

  #pragma omp critical
  {
    mycounter = *globalcounter; // handle data from mycounter to mycounter+batchsize-1
    (*globalcounter) += batchsize;
  }

  return mycounter;
}

int __cudampi__isdeviceenabled(int deviceid) {
  int val;

  #pragma omp atomic read
  val = __cudampi__deviceenabled[__cudampi__currentDevice];

  return val;
}

cudaError_t __cudampi__getDeviceCount(int *count) {
  return __cudampi_totaldevicecount;
}

cudaError_t __cudampi__cudaGetDeviceCount(int *count) {

  *count = __cudampi_totalgpudevicecount;
  return cudaSuccess;
}

cudaError_t __cudampi__cpuGetDeviceCount(int *count) {

  *count = __cudampi_totalcpudevicecount;
  return cudaSuccess;
}


cudaError_t __cudampi__getCpuFreeThreads(int* count)
{
  // TODO
  return 0;
}

void __cudampi__initializeMPI(int argc, char **argv) {

  int mtsprovided;
  int i;

  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mtsprovided);

  if (mtsprovided != MPI_THREAD_MULTIPLE) {
    printf("\nNo support for MPI_THREAD_MULTIPLE mode.\n");
    fflush(stdout);
    exit(-1);
  }

  // fetch information about the rank and number of processes

  MPI_Comm_size(MPI_COMM_WORLD, &__cudampi__MPIproccount);
  MPI_Comm_rank(MPI_COMM_WORLD, &__cudampi__myrank);

  // we assume that there are as many nodes with GPUs as the number of processes started

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

  // initialize the array -- for simplicity first try to use all available GPUs in all nodes -- query the nodes

  int localGpuCount;
  // each process first checks its own device count
  if (cudaSuccess != cudaGetDeviceCount(&localGpuCount)) {
    printf("Error invoking cudaGetDeviceCount()");
    fflush(stdout);
    exit(-1);
  }

  MPI_Allgather(&localGpuCount, 1, MPI_INT, __cudampi__GPUcountspernode, 1, MPI_INT, MPI_COMM_WORLD);

  int localCpuCount;
  // each process first checks its own device count
  if (cudaSuccess != __cudampi__getCpuFreeThreads(&localCpuCount)) {
    printf("Error invoking __cudampi__getCpuFreeThreads()");
    fflush(stdout);
    exit(-1);
  }
  // TODO: fill this variable

  MPI_Allgather(&localCpuCount, 1, MPI_INT, __cudampi__freeThreadsPerNode, 1, MPI_INT, MPI_COMM_WORLD);

  // check if there is a configuration file
  FILE *filep = fopen("__cudampi.conf", "r");

  if (filep != NULL) {
    char line[255];
    int index, val;
    while (NULL != fgets(line, 255, filep)) {
      sscanf(line, "%d:%d", &index, &val);
      __cudampi__GPUcountspernode[index] = val;
    }

    fclose(filep);
  }

// TODO also consider cpu devices
  // compute the total device count
  __cudampi_totalgpudevicecount = 0;
  for (i = 0; i < __cudampi__MPIproccount; i++) {
    __cudampi_totalgpudevicecount += __cudampi__GPUcountspernode[i];
    printf("\nOn node %d using %d GPUs.", i, __cudampi__GPUcountspernode[i]);
    if (__cudampi__freeThreadsPerNode[i] > 0 ) {
      __cudampi_totalcpudevicecount ++;
    }
    printf("\nOn node %d using %d CPU threads.", i, __cudampi__freeThreadsPerNode[i]);
  }

  __cudampi_totaldevicecount = __cudampi_totalcpudevicecount + __cudampi_totalgpudevicecount;

  printf("\n");
  fflush(stdout);
  __cudampi_localdevicecount = __cudampi__GPUcountspernode[0];

  // now compute proper indexes

  __cudampi_targetGPUfordevice = (int *)malloc(__cudampi_totaldevicecount * sizeof(int));
  if (!__cudampi_targetGPUfordevice) {
    printf("\nNot enough memory");
    exit(-1); // we could exit in a nicer way! TBD
  }


// TODO also consider cpu devices
  __cudampi_targetMPIrankfordevice = (int *)malloc(__cudampi_totaldevicecount * sizeof(int));
  if (!__cudampi_targetMPIrankfordevice) {
    printf("\nNot enough memory");
    exit(-1); // we could exit in a nicer way! TBD
  }

  int currentrank = 0;
  int currentGPU = 0;
  // now initialize values device by device
  for (i = 0; i < __cudampi_totalgpudevicecount; i++) {

  // TODO also consider cpu devices
    __cudampi_targetGPUfordevice[i] = currentGPU;
    __cudampi_targetMPIrankfordevice[i] = currentrank;

    __cudampi__devicepower[i] = -1; // initial value
    __cudampi__deviceenabled[i] = 1;

    currentGPU++;

    if (currentGPU == __cudampi__GPUcountspernode[currentrank]) {
      // reset the GPU id and increase the rank
      currentGPU = 0;
      currentrank++;
    }
  }

  currentrank = 0;
  for (i = __cudampi_totalgpudevicecount; i < __cudampi_totaldevicecount; i++) {

  // TODO also consider cpu devices
    __cudampi_targetGPUfordevice[i] = -1;
    while (__cudampi__freeThreadsPerNode[i] <= 0)
    {
      // skip all nodes with no free threads;
      currentrank ++;
    }
    __cudampi_targetMPIrankfordevice[i] = currentrank;
  }
  // initialize current device id to 0 although various threads are expected to have various GPU ids

  for (int i = 0; i < __CUDAMPI_MAX_THREAD_COUNT; i++) {
    __cudampi__currentdevice[i] = 0;

    // initialize locks as well
    omp_init_lock(&(__cudampi__devicelocks[i]));
  }

  omp_init_lock(&deviceselectionlock);

  __cudampi__amimanager[0] = 1;

  MPI_Bcast(&__cudampi_totaldevicecount, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Bcast(__cudampi_targetMPIrankfordevice, __cudampi_totaldevicecount, MPI_INT, 0, MPI_COMM_WORLD);

  // set up communicators - there should be a communicator for each target GPU, each of which is shared by process 0 and process on which the target GPU resides, this is obviously not needed for local GPUs

  __cudampi__communicators = (MPI_Comm *)malloc(sizeof(MPI_Comm) * __cudampi_totaldevicecount);
  if (!__cudampi__communicators) {
    printf("\nNot enough memory for communicators");
    exit(-1); // we could exit in a nicer way! TBD
  }

  for (int i = __cudampi__GPUcountspernode[0]; i < __cudampi_totaldevicecount; i++) { // disregard local GPUs since we do not need communicators for them -- communication will be by direct invocations

    int ranks[2] = {0, __cudampi_targetMPIrankfordevice[i]}; // group and communicator between process 0 and the process of the target GPU/device

    MPI_Group groupall;
    MPI_Comm_group(MPI_COMM_WORLD, &groupall);

    // Keep only process 0 and the process handling the given GPU
    MPI_Group tempgroup;
    MPI_Group_incl(groupall, 2, ranks, &tempgroup);

    MPI_Comm_create(MPI_COMM_WORLD, tempgroup, &(__cudampi__communicators[i]));
  }
}

void __cudampi__terminateMPI() {

  // finalize the other nodes -> shut down threads responsible for remote GPUs

  for (int i = __cudampi_localdevicecount; i < __cudampi_totaldevicecount; i++) {
    MPI_Send(NULL, 0, MPI_CHAR, 1 /*__cudampi__gettargetMPIrank(i)*/, __cudampi__CUDAMPIFINALIZE, __cudampi__communicators[i]);
  }

  MPI_Finalize();
}

int __cudampi__gettargetGPU(int device) {
  // gets target GPU id

  return __cudampi_targetGPUfordevice[device];
  //  return device%__cudampi__GPUcountpernode;
}

int __cudampi__gettargetMPIrank(int device) {
  // gets target MPI rank based on local GPU id

  return __cudampi_targetMPIrankfordevice[device];
  //  return device/__cudampi__GPUcountpernode;
}

cudaError_t __cudampi__cudaMalloc(void **devPtr, size_t size) {

  if (__cudampi_isLocalGpu) { // run locally
    return cudaMalloc(devPtr, size);
  } else { // allocate remotely

    // we then return the actual pointer from another node -- it is used only on that node

    // request allocation on the other node

    int targetrank = __cudampi__gettargetMPIrank(__cudampi__currentDevice);

    unsigned long sdata = size; // how many bytes to allocate on the GPU

    MPI_Send((void *)(&sdata), 1, MPI_UNSIGNED_LONG, 1 /*targetrank*/, __cudampi__CUDAMPIMALLOCREQ, __cudampi__currentCommunicator);

    int rsize = sizeof(void *) + sizeof(cudaError_t);
    // receive confirmation with the actual pointer
    unsigned char rdata[rsize];

    MPI_Recv(rdata, rsize, MPI_UNSIGNED_CHAR, 1 /*targetrank*/, __cudampi__CUDAMPIMALLOCRESP, __cudampi__currentCommunicator, NULL);

    *devPtr = *((void **)rdata);

    return ((cudaError_t)(rdata + sizeof(void *)));
  }
}
cudaError_t __cudampi__cpuMalloc(void **devPtr, size_t size) {
  // TODO
  return cudaSuccess;
}

cudaError_t __cudampi__malloc(void **devPtr, size_t size) {
  if (__cudampi__isCpu()) {
    return __cudampi__cpuMalloc(devPtr, size);
  }
  // else
  return __cudampi__cudaMalloc(devPtr, size);
}


cudaError_t __cudampi__cudaFree(void *devPtr) {
  // as for cudaMalloc but just free

  if (__cudampi_isLocalGpu) { // run locally
    return cudaFree(devPtr);
  } else { // allocate remotely

    int targetrank = __cudampi__gettargetMPIrank(__cudampi__currentDevice);

    int ssize = sizeof(void *);
    unsigned char sdata[ssize];

    *((void **)sdata) = devPtr;

    MPI_Send((void *)sdata, ssize, MPI_UNSIGNED_CHAR, 1 /*targetrank*/, __cudampi__CUDAMPIFREEREQ, __cudampi__currentCommunicator);

    int rsize = sizeof(cudaError_t);
    unsigned char rdata[rsize];

    MPI_Recv(rdata, rsize, MPI_UNSIGNED_CHAR, 1 /*targetrank*/, __cudampi__CUDAMPIFREERESP, __cudampi__currentCommunicator, NULL);

    return ((cudaError_t)rdata);
  }
}

cudaError_t __cudampi__cudaDeviceSynchronize(void)
{
  return __cudampi__deviceSynchronize();
}

cudaError_t __cudampi__deviceSynchronize(void) {

  cudaError_t retVal;
  static int selecteddevices = 0; // only updated by thread 0
  int amimanager;                 // if the current thread is manager for device selection

  // if ((powermeasurecounter[omp_get_thread_num()]%10)==4) {

  omp_set_lock(&(__cudampi__devicelocks[__cudampi__currentDevice]));
  amimanager = __cudampi__amimanager[__cudampi__currentDevice];
  omp_unset_lock(&(__cudampi__devicelocks[__cudampi__currentDevice]));

  if (amimanager) {
    if (__cudampi__isglobalpowerlimitset) {
      if (!selecteddevices) {
        selecteddevices = __cudampi__selectdevicesforpowerlimit_greedy();
      } else {

        // adjust if needed
        float power;
        power = __cudampi__gettotalpowerofselecteddevices();
        if (power != (-1)) {
          if (power > __cudampi__globalpowerlimit) {
            printf("\ntotal power=%f limit=%f, adjusting", power, __cudampi__globalpowerlimit);
            fflush(stdout);
            __cudampi__selectdevicesforpowerlimit_greedy();
          }
        }
      }
    }
  }
  if (__cudampi__isLocalCpu) { // run CPU synchronization locally
    // TODO
    // compute power
    // wait for computations to finish
  } else if (__cudampi__isCpu()) { // run CPU synchronization remotely
    // TODO
  } else if (__cudampi_isLocalGpu) { // run GPU synchronization locally

    // now get power measurement - this should be OK as we assume that computations might be taking place

    //   if ((powermeasurecounter[omp_get_thread_num()]%10)==4) {

    omp_set_lock(&(__cudampi__devicelocks[__cudampi__currentDevice]));
    __cudampi__devicepower[__cudampi__currentDevice] = getGPUpower(__cudampi__currentDevice);
    omp_unset_lock(&(__cudampi__devicelocks[__cudampi__currentDevice]));

    retVal = cudaDeviceSynchronize();
  } else { // run GPU synchronization remotely

    int targetrank = __cudampi__gettargetMPIrank(__cudampi__currentDevice);

    int sdata; // if 0 then means do not measure power, if 1 do measure on the slave side
               //    if ((powermeasurecounter[omp_get_thread_num()]%10)==4)
    sdata = 1;
    //    else sdata=0;

    MPI_Send(&sdata, 1, MPI_INT, 1 /*targetrank*/, __cudampi__CUDAMPIDEVICESYNCHRONIZEREQ, __cudampi__currentCommunicator);

    // receive an error message and a float representing power consumption

    int rsize = sizeof(cudaError_t) + sizeof(float);
    unsigned char rdata[rsize];

    MPI_Recv(rdata, rsize, MPI_UNSIGNED_CHAR, 1 /*targetrank*/, __cudampi__CUDAMPIDEVICESYNCHRONIZERESP, __cudampi__currentCommunicator, NULL);

    // decode and store power consumption for the device

    float power = *((float *)(rdata + sizeof(cudaError_t)));

    if (power != (-1)) {
      omp_set_lock(&(__cudampi__devicelocks[__cudampi__currentDevice]));
      __cudampi__devicepower[__cudampi__currentDevice] = power;
      omp_unset_lock(&(__cudampi__devicelocks[__cudampi__currentDevice]));
    }

    retVal = ((cudaError_t)rdata);
  }

  powermeasurecounter[omp_get_thread_num()]++;

  // record time
  if (__cudampi__timemeasured[__cudampi__currentDevice]) {
    gettimeofday(&(__cudampi__timestop[__cudampi__currentDevice]), NULL);

    omp_set_lock(&(__cudampi__devicelocks[__cudampi__currentDevice]));
    __cudampi__time[__cudampi__currentDevice].tv_sec = __cudampi__timestop[__cudampi__currentDevice].tv_sec - __cudampi__timestart[__cudampi__currentDevice].tv_sec;    // compute current time
    __cudampi__time[__cudampi__currentDevice].tv_usec = __cudampi__timestop[__cudampi__currentDevice].tv_usec - __cudampi__timestart[__cudampi__currentDevice].tv_usec; // compute current time
    omp_unset_lock(&(__cudampi__devicelocks[__cudampi__currentDevice]));

    __cudampi__timestart[__cudampi__currentDevice] = __cudampi__timestop[__cudampi__currentDevice];

  } else {
    __cudampi__timemeasured[__cudampi__currentDevice] = 1;
    gettimeofday(&(__cudampi__timestart[__cudampi__currentDevice]), NULL);
  }

  return retVal;
}

cudaError_t __cudampi__cudaSetDevice(int device) {

  __cudampi__currentDevice = device; // set it for the current thread

  if (__cudampi_isLocalGpu) { // run locally
    return cudaSetDevice(device);
  } else { // set device remotely

    int targetrank = __cudampi__gettargetMPIrank(__cudampi__currentDevice);

    int sdata = __cudampi__gettargetGPU(__cudampi__currentDevice); // compute the target GPU id

    MPI_Send(&sdata, 1, MPI_INT, 1 /*targetrank*/, __cudampi__CUDAMPISETDEVICEREQ, __cudampi__currentCommunicator);

    int rsize = sizeof(cudaError_t);
    unsigned char rdata[rsize];

    MPI_Recv(rdata, rsize, MPI_UNSIGNED_CHAR, 1 /*targetrank*/, __cudampi__CUDAMPISETDEVICERESP, __cudampi__currentCommunicator, NULL);

    return ((cudaError_t)rdata);
  }
}

int __cudampi__isCpu()
{
  // TODO
  return __cudampi__currentDevice  >= __cudampi_totalgpudevicecount;
}

cudaError_t __cudampi__setDevice(int device) {
  __cudampi__currentDevice = device; // set it for the current thread
  if (!__cudampi__isCpu()) {
    return __cudampi__cudaSetDevice(device);
  }
  // else
  return cudaSuccess;
}

cudaError_t __cudampi__cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {

  if (__cudampi_isLocalGpu) { // run locally
    return cudaMemcpy(dst, src, count, kind);
  } else if (kind == cudaMemcpyHostToDevice) {

    int targetrank = __cudampi__gettargetMPIrank(__cudampi__currentDevice);

    size_t ssize = sizeof(void *) + count;
    unsigned char sdata[ssize];

    *((void **)sdata) = dst;
    memcpy(sdata + sizeof(void *), src, count); // copy input data

    MPI_Send((void *)sdata, ssize, MPI_UNSIGNED_CHAR, 1 /*targetrank*/, __cudampi__CUDAMPIHOSTTODEVICEREQ, __cudampi__currentCommunicator);

    int rsize = sizeof(cudaError_t);
    unsigned char rdata[rsize];

    MPI_Recv(rdata, rsize, MPI_UNSIGNED_CHAR, 1 /*targetrank*/, __cudampi__CUDAMPIHOSTTODEVICERESP, __cudampi__currentCommunicator, NULL);

    return ((cudaError_t)rdata);

  } else if (kind == cudaMemcpyDeviceToHost) {

    int targetrank = __cudampi__gettargetMPIrank(__cudampi__currentDevice);

    size_t ssize = sizeof(void *) + sizeof(unsigned long);
    unsigned char sdata[ssize];

    *((void **)sdata) = (void *)src;
    *((unsigned long *)(sdata + sizeof(void *))) = count; // how many bytes we want to get from a GPU

    MPI_Send((void *)sdata, ssize, MPI_UNSIGNED_CHAR, 1 /*targetrank*/, __cudampi__CUDAMPIDEVICETOHOSTREQ, __cudampi__currentCommunicator);

    size_t rsize = sizeof(cudaError_t) + count;
    unsigned char rdata[rsize];

    MPI_Recv(rdata, rsize, MPI_UNSIGNED_CHAR, 1 /*targetrank*/, __cudampi__CUDAMPIDEVICETOHOSTRESP, __cudampi__currentCommunicator, NULL);

    memcpy(dst, rdata + sizeof(cudaError_t), count);

    return ((cudaError_t)rdata);
  }
}

cudaError_t __cudampi__cpuMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
  // TODO
  return cudaSuccess;
}

cudaError_t __cudampi__cudaMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream) {

  if (__cudampi_isLocalGpu) { // run locally
    return cudaMemcpyAsync(dst, src, count, kind, stream);
  } else if (kind == cudaMemcpyHostToDevice) {

    int targetrank = __cudampi__gettargetMPIrank(__cudampi__currentDevice);

    size_t ssize = sizeof(void *) + sizeof(cudaStream_t) + count;
    unsigned char sdata[ssize];

    *((void **)sdata) = dst;
    *((cudaStream_t *)(sdata + sizeof(void *))) = stream;
    memcpy(sdata + sizeof(void *) + sizeof(cudaStream_t), src, count); // copy input data

    MPI_Send((void *)sdata, ssize, MPI_UNSIGNED_CHAR, 1 /*targetrank*/, __cudampi__CUDAMPIHOSTTODEVICEASYNCREQ, __cudampi__currentCommunicator);

    int rsize = sizeof(cudaError_t);
    unsigned char rdata[rsize];

    MPI_Recv(rdata, rsize, MPI_UNSIGNED_CHAR, 1 /*targetrank*/, __cudampi__CUDAMPIHOSTTODEVICEASYNCRESP, __cudampi__currentCommunicator, NULL);

    return ((cudaError_t)rdata);

  } else if (kind == cudaMemcpyDeviceToHost) {

    int targetrank = __cudampi__gettargetMPIrank(__cudampi__currentDevice);

    size_t ssize = sizeof(void *) + sizeof(unsigned long) + sizeof(cudaStream_t);
    unsigned char sdata[ssize];

    *((void **)sdata) = (void *)src;
    *((unsigned long *)(sdata + sizeof(void *))) = count; // how many bytes we want to get from a GPU
    *((cudaStream_t *)(sdata + sizeof(void *) + sizeof(unsigned long))) = stream;

    MPI_Send((void *)sdata, ssize, MPI_UNSIGNED_CHAR, 1 /*targetrank*/, __cudampi__CUDAMPIDEVICETOHOSTASYNCREQ, __cudampi__currentCommunicator);

    size_t rsize = sizeof(cudaError_t) + count;
    unsigned char rdata[rsize];

    MPI_Recv(rdata, rsize, MPI_UNSIGNED_CHAR, 1 /*targetrank*/, __cudampi__CUDAMPIDEVICETOHOSTASYNCRESP, __cudampi__currentCommunicator, NULL);

    memcpy(dst, rdata + sizeof(cudaError_t), count);

    return ((cudaError_t)rdata);
  }
}

cudaError_t __cudampi__cpuMemcpyAsync(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
  return cudaSuccess;
}

void launchkernelinstream(void *devPtr, cudaStream_t stream);

void __cudampi__kernelinstream(void *devPtr, cudaStream_t stream) {

  if (__cudampi_isLocalGpu) { // run locally
    launchkernelinstream(devPtr, stream);
  } else { // launch remotely

    int targetrank = __cudampi__gettargetMPIrank(__cudampi__currentDevice);

    size_t ssize = sizeof(void *) + sizeof(cudaStream_t);
    unsigned char sdata[ssize];

    *((void **)sdata) = devPtr;
    *((cudaStream_t *)(sdata + sizeof(void *))) = stream;
    MPI_Send((void *)sdata, ssize, MPI_UNSIGNED_CHAR, 1 /*targetrank*/, __cudampi__CUDAMPILAUNCHKERNELINSTREAMREQ, __cudampi__currentCommunicator);

    int rsize = sizeof(cudaError_t);
    unsigned char rdata[rsize];

    MPI_Recv(rdata, rsize, MPI_UNSIGNED_CHAR, 1 /*targetrank*/, __cudampi__CUDAMPILAUNCHKERNELINSTREAMRESP, __cudampi__currentCommunicator, NULL);
  }
}

void launchkernel(void *devPtr);

void __cudampi__kernel(void *devPtr) {

  if (__cudampi_isLocalGpu) { // run locally
    launchkernel(devPtr);
  } else { // launch remotely

    int targetrank = __cudampi__gettargetMPIrank(__cudampi__currentDevice);

    size_t ssize = sizeof(void *);
    unsigned char sdata[ssize];

    *((void **)sdata) = devPtr;

    MPI_Send((void *)sdata, ssize, MPI_UNSIGNED_CHAR, 1 /*targetrank*/, __cudampi__CUDAMPILAUNCHKERNELREQ, __cudampi__currentCommunicator);

    int rsize = sizeof(cudaError_t);
    unsigned char rdata[rsize];

    MPI_Recv(rdata, rsize, MPI_UNSIGNED_CHAR, 1 /*targetrank*/, __cudampi__CUDAMPILAUNCHKERNELRESP, __cudampi__currentCommunicator, NULL);
  }
}

void cpukernel(void *devPtr, int num_threads);

void __cudampi__cpuKernel(void *devPtr){

  if (__cudampi__isLocalCpu) { // run locally
    cpukernel(devPtr, __cudampi__freeThreadsPerNode[0]);
  } else { // launch remotely
    // TODO
  }
}

cudaError_t __cudampi__cudaStreamCreate(cudaStream_t *pStream) {

  if (__cudampi_isLocalGpu) { // run locally
    return cudaStreamCreate(pStream);
  } else { // create a stream remotely

    // we then return the actual pointer from another node -- it is used only on that node

    int targetrank = __cudampi__gettargetMPIrank(__cudampi__currentDevice);

    // send an empty message -- there is no need for input data
    MPI_Send(NULL, 0, MPI_UNSIGNED_LONG, 1 /*targetrank*/, __cudampi__CUDAMPISTREAMCREATEREQ, __cudampi__currentCommunicator);

    // get a pointer to the newly created stream + an error message
    int rsize = sizeof(cudaStream_t) + sizeof(cudaError_t);
    // receive confirmation with the actual pointer
    unsigned char rdata[rsize];

    MPI_Recv(rdata, rsize, MPI_UNSIGNED_CHAR, 1 /*targetrank*/, __cudampi__CUDAMPISTREAMCREATERESP, __cudampi__currentCommunicator, NULL);

    *pStream = *((cudaStream_t *)rdata);

    return ((cudaError_t)(rdata + sizeof(void *)));
  }
}

cudaError_t __cudampi__cudaStreamDestroy(cudaStream_t stream) {

  if (__cudampi_isLocalGpu) { // run locally
    return cudaStreamDestroy(stream);
  } else { // destroy remotely

    int targetrank = __cudampi__gettargetMPIrank(__cudampi__currentDevice);

    int ssize = sizeof(cudaStream_t);
    unsigned char sdata[ssize];

    *((cudaStream_t *)sdata) = stream;

    MPI_Send((unsigned char *)sdata, ssize, MPI_UNSIGNED_CHAR, 1 /*targetrank*/, __cudampi__CUDAMPISTREAMDESTROYREQ, __cudampi__currentCommunicator);

    int rsize = sizeof(cudaError_t);
    unsigned char rdata[rsize];

    MPI_Recv(rdata, rsize, MPI_UNSIGNED_CHAR, 1 /*targetrank*/, __cudampi__CUDAMPISTREAMDESTROYRESP, __cudampi__currentCommunicator, NULL);

    return ((cudaError_t)rdata);
  }
}
