/*
Copyright 2023 Paweł Czarnul pczarnul@eti.pg.edu.pl

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#include "cudampilib.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include <sys/time.h>

long long VECTORSIZE = 200000000;

double *vectora;
double *vectorc;

int batchsize = 50000; // 10000;

long long globalcounter = 0; // access to it controlled within a critical section

int streamcount = 1;
float powerlimit;

int main(int argc, char **argv) {

  struct timeval start, stop;
  struct timeval starttotal, stoptotal;

  gettimeofday(&starttotal, NULL);

  __cudampi__initializeMPI(argc, argv);

  int alldevicescount = 0;

  if (argc > 1) {
    streamcount = atoi(argv[1]);
  }

  if (argc > 2) {
    powerlimit = atof(argv[2]);
    printf("\nSetting power limit=%f\n", powerlimit);
    fflush(stdout);
    __cudampi__setglobalpowerlimit(powerlimit);
  }

  __cudampi__getDeviceCount(&alldevicescount);

  cudaHostAlloc((void **)&vectora, sizeof(double) * VECTORSIZE, cudaHostAllocDefault);
  if (!vectora) {
    printf("\nNot enough memory.");
    exit(0);
  }

  cudaHostAlloc((void **)&vectorc, sizeof(double) * VECTORSIZE, cudaHostAllocDefault);
  if (!vectorc) {
    printf("\nNot enough memory.");
    exit(0);
  }

  for (long long i = 0; i < VECTORSIZE; i++) {
    vectora[i] = (80000000 + i) % 2000000000;
  }

  gettimeofday(&start, NULL);

  #pragma omp parallel num_threads(alldevicescount)
  {
    long long mycounter;
    int finish = 0;
    void *devPtra, *devPtrc;
    void *devPtra2, *devPtrc2;
    int i;
    cudaStream_t stream1;
    cudaStream_t stream2;
    int mythreadid = omp_get_thread_num();
    void *devPtr;
    void *devPtr2;
    long long privatecounter = 0;
    __cudampi__setDevice(mythreadid);
    #pragma omp barrier
    __cudampi__malloc(&devPtra, batchsize * sizeof(double));
    __cudampi__malloc(&devPtrc, batchsize * sizeof(double));

    __cudampi__malloc(&devPtr, 2 * sizeof(void *));

    __cudampi__malloc(&devPtra2, batchsize * sizeof(double));
    __cudampi__malloc(&devPtrc2, batchsize * sizeof(double));

    __cudampi__malloc(&devPtr2, 2 * sizeof(void *));

    
    __cudampi__streamCreate(&stream1);
    __cudampi__streamCreate(&stream2);

    __cudampi__memcpyAsync(devPtr, &devPtra, sizeof(void *), cudaMemcpyHostToDevice, stream1);
    __cudampi__memcpyAsync(devPtr + sizeof(void *), &devPtrc, sizeof(void *), cudaMemcpyHostToDevice, stream1);

    __cudampi__memcpyAsync(devPtr2, &devPtra2, sizeof(void *), cudaMemcpyHostToDevice, stream2);
    __cudampi__memcpyAsync(devPtr2 + sizeof(void *), &devPtrc2, sizeof(void *), cudaMemcpyHostToDevice, stream2); 

    do {

      mycounter = __cudampi__getnextchunkindex(&globalcounter, batchsize, VECTORSIZE);

      if (mycounter >= VECTORSIZE) {
        finish = 1;
      } 
      else {
        __cudampi__memcpyAsync(devPtra, vectora + mycounter, batchsize * sizeof(double), cudaMemcpyHostToDevice, stream1);

        __cudampi__kernelInStream(devPtr, stream1);

        __cudampi__memcpyAsync(vectorc + mycounter, devPtrc, batchsize * sizeof(double), cudaMemcpyDeviceToHost, stream1);

        // do it again in the second stream
        if (streamcount == 2) {
          mycounter = __cudampi__getnextchunkindex(&globalcounter, batchsize, VECTORSIZE);

          if (mycounter >= VECTORSIZE) {
            finish = 1;
          } else {

            __cudampi__memcpyAsync(devPtra2, vectora + mycounter, batchsize * sizeof(double), cudaMemcpyHostToDevice, stream2);

            __cudampi__kernelInStream(devPtr2, stream2);

            __cudampi__memcpyAsync(vectorc + mycounter, devPtrc2, batchsize * sizeof(double), cudaMemcpyDeviceToHost, stream2);
          }
        }
      }

      privatecounter++;
      if (privatecounter % 2) {
        __cudampi__deviceSynchronize();
      }

    } while (!finish);

    __cudampi__deviceSynchronize();
    __cudampi__streamDestroy(stream1);
    __cudampi__streamDestroy(stream2);
  }

  gettimeofday(&stop, NULL);

  printf("Main elapsed time=%f\n", (double)((stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) / 1000000.0));
  fflush(stdout);

  __cudampi__terminateMPI();

  gettimeofday(&stoptotal, NULL);
  printf("Total elapsed time=%f\n", (double)((stoptotal.tv_sec - starttotal.tv_sec) + (double)(stoptotal.tv_usec - starttotal.tv_usec) / 1000000.0));
  fflush(stdout);
}