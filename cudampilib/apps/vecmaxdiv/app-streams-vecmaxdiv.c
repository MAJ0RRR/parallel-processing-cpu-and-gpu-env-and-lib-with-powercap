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
// #define VECTORSIZE 120000000

double *vectora;
double *vectorb;
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

  int cudadevicescount = 1;

  if (argc > 1) {
    streamcount = atoi(argv[1]);
  }

  if (argc > 2) {
    powerlimit = atof(argv[2]);
    printf("\nSetting power limit=%f\n", powerlimit);
    fflush(stdout);
    __cudampi__setglobalpowerlimit(powerlimit);
  }
  //  printf("\nStream count %d",streamcount);

  __cudampi__cudaGetDeviceCount(&cudadevicescount);

  cudaHostAlloc((void **)&vectora, sizeof(double) * VECTORSIZE, cudaHostAllocDefault);
  if (!vectora) {
    printf("\nNot enough memory.");
    exit(0);
  }

  cudaHostAlloc((void **)&vectorb, sizeof(double) * VECTORSIZE, cudaHostAllocDefault);
  if (!vectorb) {
    printf("\nNot enough memory.");
    exit(0);
  }

  cudaHostAlloc((void **)&vectorc, sizeof(double) * VECTORSIZE, cudaHostAllocDefault);
  if (!vectorc) {
    printf("\nNot enough memory.");
    exit(0);
  }

  for (long long i = 0; i < VECTORSIZE; i++) {
    vectora[i] = 2 * ((200000000 + i) % 1000000000) + 1;
    vectorb[i] = 2 * ((200000000 + i) % 1000000000) + 3;
  }

  gettimeofday(&start, NULL);

  #pragma omp parallel num_threads(cudadevicescount)
  {

    long long mycounter;
    int finish = 0;
    void *devPtra, *devPtrb, *devPtrc;
    void *devPtra2, *devPtrb2, *devPtrc2;
    int i;
    cudaStream_t stream1;
    cudaStream_t stream2;
    int mythreadid = omp_get_thread_num();
    void *devPtr;
    void *devPtr2;
    long long privatecounter = 0;

    //   for(i=0;i<2048;i++)
    //   printf("\n ind[%d]=%d",i,(int)(tab[i]));

    __cudampi__cudaSetDevice(mythreadid);
    #pragma omp barrier
    __cudampi__cudaMalloc(&devPtra, batchsize * sizeof(double));
    __cudampi__cudaMalloc(&devPtrb, batchsize * sizeof(double));
    __cudampi__cudaMalloc(&devPtrc, batchsize * sizeof(double));

    __cudampi__cudaMalloc(&devPtr, 3 * sizeof(void *));

    __cudampi__cudaMalloc(&devPtra2, batchsize * sizeof(double));
    __cudampi__cudaMalloc(&devPtrb2, batchsize * sizeof(double));
    __cudampi__cudaMalloc(&devPtrc2, batchsize * sizeof(double));

    __cudampi__cudaMalloc(&devPtr2, 3 * sizeof(void *));

    __cudampi__cudaStreamCreate(&stream1);

    __cudampi__cudaStreamCreate(&stream2);

    __cudampi__cudaMemcpyAsync(devPtr, &devPtra, sizeof(void *), cudaMemcpyHostToDevice, stream1);
    __cudampi__cudaMemcpyAsync(devPtr + sizeof(void *), &devPtrb, sizeof(void *), cudaMemcpyHostToDevice, stream1);
    __cudampi__cudaMemcpyAsync(devPtr + 2 * sizeof(void *), &devPtrc, sizeof(void *), cudaMemcpyHostToDevice, stream1);

    __cudampi__cudaMemcpyAsync(devPtr2, &devPtra2, sizeof(void *), cudaMemcpyHostToDevice, stream2);
    __cudampi__cudaMemcpyAsync(devPtr2 + sizeof(void *), &devPtrb2, sizeof(void *), cudaMemcpyHostToDevice, stream2);
    __cudampi__cudaMemcpyAsync(devPtr2 + 2 * sizeof(void *), &devPtrc2, sizeof(void *), cudaMemcpyHostToDevice, stream2);

    //    exit(0);

    /*  printf("pointer=%p in client",devPtr);
    fflush(stdout);
    */

    //  for(int j=0;j<1;j++) {
    do {

      /*
#pragma omp critical
{
  mycounter=globalcounter; // handle data from mycounter to mycounter+batchsize-1
  globalcounter+=batchsize;


  //printf("\nthread=%d counter=%d",mythreadid,mycounter);
  //fflush(stdout);


}
*/
      mycounter = __cudampi__getnextchunkindex(&globalcounter, batchsize, VECTORSIZE);

      if (mycounter >= VECTORSIZE) {
        finish = 1;
      } else {

        __cudampi__cudaMemcpyAsync(devPtra, vectora + mycounter, batchsize * sizeof(double), cudaMemcpyHostToDevice, stream1);
        __cudampi__cudaMemcpyAsync(devPtrb, vectorb + mycounter, batchsize * sizeof(double), cudaMemcpyHostToDevice, stream1);

        __cudampi__kernelinstream(devPtr, stream1);

        __cudampi__cudaMemcpyAsync(vectorc + mycounter, devPtrc, batchsize * sizeof(double), cudaMemcpyDeviceToHost, stream1);
      }
      // do it again in the second stream
      if (streamcount == 2) {
        if (!finish) {

          /*
          #pragma omp critical
              {
                mycounter=globalcounter; // handle data from mycounter to mycounter+batchsize-1
                globalcounter+=batchsize;



              }
            */

          mycounter = __cudampi__getnextchunkindex(&globalcounter, batchsize, VECTORSIZE);

          if (mycounter >= VECTORSIZE) {
            finish = 1;
          } else {

            __cudampi__cudaMemcpyAsync(devPtra2, vectora + mycounter, batchsize * sizeof(double), cudaMemcpyHostToDevice, stream2);
            __cudampi__cudaMemcpyAsync(devPtrb2, vectorb + mycounter, batchsize * sizeof(double), cudaMemcpyHostToDevice, stream2);

            __cudampi__kernelinstream(devPtr2, stream2);

            __cudampi__cudaMemcpyAsync(vectorc + mycounter, devPtrc2, batchsize * sizeof(double), cudaMemcpyDeviceToHost, stream2);
          }
        }
      }

      privatecounter++;
      if (privatecounter % 2) {
        __cudampi__cudaDeviceSynchronize();
        //    printf("\nthread %d sync mycounter=%d",omp_get_thread_num(),mycounter);
        //     fflush(stdout);
      }

    } while (!finish);

    /*
    finish=0;
  #pragma omp barrier

    #pragma omp single
    {
      globalcounter=0;

    }
    }
    */

    __cudampi__cudaDeviceSynchronize();

    __cudampi__cudaStreamDestroy(stream1);

    //  __cudampi__cudaFree(devPtr);
  }

  gettimeofday(&stop, NULL);

  printf("Main elapsed time=%f\n", (double)((stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) / 1000000.0));
  fflush(stdout);

  __cudampi__terminateMPI();

  gettimeofday(&stoptotal, NULL);
  printf("Total elapsed time=%f\n", (double)((stoptotal.tv_sec - starttotal.tv_sec) + (double)(stoptotal.tv_usec - starttotal.tv_usec) / 1000000.0));
  fflush(stdout);
}
