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

#define ENABLE_LOGGING
#include "logger.h"
#include "vecadd_defines.h"

#define VECTORSIZE VECADD_VECTOR_SIZE

double *vectora;
double *vectorb;
double *vectorc;

int batchsize = VECADD_BATCH_SIZE;

long long globalcounter = 0;

int streamcount = 1;
float powerlimit = 0;

int main(int argc, char **argv) 
{
  struct timeval start, stop;
  struct timeval starttotal, stoptotal;

  gettimeofday(&starttotal, NULL);

  __cudampi__initializeMPI(argc, argv);

  int alldevicescount = 0;

  if (argc > 1) 
  {
    streamcount = atoi(argv[1]);
  }

  if (argc > 2) 
  {
    powerlimit = atof(argv[2]);
    log_message(LOG_INFO, "\nSetting power limit=%f\n", powerlimit);
    __cudampi__setglobalpowerlimit(powerlimit);
  }

  __cudampi__getDeviceCount(&alldevicescount);

  cudaHostAlloc((void **)&vectora, sizeof(double) * VECTORSIZE, cudaHostAllocDefault);
  if (!vectora) 
  {
    log_message(LOG_ERROR, "\nNot enough memory.");
    exit(0);
  }

  cudaHostAlloc((void **)&vectorb, sizeof(double) * VECTORSIZE, cudaHostAllocDefault);
  if (!vectorb) 
  {
    log_message(LOG_ERROR, "\nNot enough memory.");
    exit(0);
  }

  cudaHostAlloc((void **)&vectorc, sizeof(double) * VECTORSIZE, cudaHostAllocDefault);
  if (!vectorc) 
  {
    log_message(LOG_ERROR, "\nNot enough memory.");
    exit(0);
  }

  gettimeofday(&start, NULL);

#pragma omp parallel num_threads(alldevicescount)
{
    struct timeval start_p, stop_p;
    struct timeval alloc_start, alloc_end;
    struct timeval index_start, index_end;
    struct timeval memcpy_start, memcpy_end;
    struct timeval kernel_start, kernel_end;
    struct timeval sync_start, sync_end;
    double time_alloc = 0.0;
    double time_index = 0.0;
    double time_memcpy = 0.0;
    double time_kernel = 0.0;
    double time_sync = 0.0;
    int mycounter;
    int counterrr = 0;
    int finish = 0;
    void *devPtra, *devPtrb, *devPtrc;
    void *devPtra2, *devPtrb2, *devPtrc2;
    int i;
    cudaStream_t stream1;
    cudaStream_t stream2;
    int mythreadid = omp_get_thread_num();
    void *devPtr;
    void *devPtr2;
    int privatecounter = 0;

    __cudampi__setDevice(mythreadid);

    #pragma omp barrier

    gettimeofday(&start_p, NULL);

    // Timing the memory allocation block
    gettimeofday(&alloc_start, NULL);

    __cudampi__malloc(&devPtra, batchsize * sizeof(double));
    __cudampi__malloc(&devPtrb, batchsize * sizeof(double));
    __cudampi__malloc(&devPtrc, batchsize * sizeof(double));

    __cudampi__malloc(&devPtr, 3 * sizeof(void *));

    if(streamcount == 2)
    {
        __cudampi__malloc(&devPtra2, batchsize * sizeof(double));
        __cudampi__malloc(&devPtrb2, batchsize * sizeof(double));
        __cudampi__malloc(&devPtrc2, batchsize * sizeof(double));

        __cudampi__malloc(&devPtr2, 3 * sizeof(void *));
    }

    gettimeofday(&alloc_end, NULL);
    time_alloc = (alloc_end.tv_sec - alloc_start.tv_sec) + ((alloc_end.tv_usec - alloc_start.tv_usec)/1e6);

    __cudampi__streamCreate(&stream1);
    if(streamcount == 2)
    {
        __cudampi__streamCreate(&stream2);

        // Timing memcpy operations for devPtr2
        gettimeofday(&memcpy_start, NULL);
        __cudampi__memcpyAsync(devPtr2, &devPtra2, sizeof(void *), cudaMemcpyHostToDevice, stream2);
        __cudampi__memcpyAsync(devPtr2 + sizeof(void *), &devPtrb2, sizeof(void *), cudaMemcpyHostToDevice, stream2);
        __cudampi__memcpyAsync(devPtr2 + 2 * sizeof(void *), &devPtrc2, sizeof(void *), cudaMemcpyHostToDevice, stream2);
        gettimeofday(&memcpy_end, NULL);
        time_memcpy += (memcpy_end.tv_sec - memcpy_start.tv_sec) + ((memcpy_end.tv_usec - memcpy_start.tv_usec)/1e6);
    }

    // Timing memcpy operations for devPtr
    gettimeofday(&memcpy_start, NULL);
    __cudampi__memcpyAsync(devPtr, &devPtra, sizeof(void *), cudaMemcpyHostToDevice, stream1);
    __cudampi__memcpyAsync(devPtr + sizeof(void *), &devPtrb, sizeof(void *), cudaMemcpyHostToDevice, stream1);
    __cudampi__memcpyAsync(devPtr + 2 * sizeof(void *), &devPtrc, sizeof(void *), cudaMemcpyHostToDevice, stream1);
    gettimeofday(&memcpy_end, NULL);
    time_memcpy += (memcpy_end.tv_sec - memcpy_start.tv_sec) + ((memcpy_end.tv_usec - memcpy_start.tv_usec)/1e6);

    do 
    {
        // Timing the index fetching block
        gettimeofday(&index_start, NULL);
        mycounter = __cudampi__getnextchunkindex(&globalcounter, batchsize, VECTORSIZE);
        counterrr += 1;
        gettimeofday(&index_end, NULL);
        time_index += (index_end.tv_sec - index_start.tv_sec) + ((index_end.tv_usec - index_start.tv_usec)/1e6);

        if (mycounter >= VECTORSIZE) 
        {
            finish = 1;
        }
        else
        {
            // Timing memcpy operations
            gettimeofday(&memcpy_start, NULL);
            __cudampi__memcpyAsync(devPtra, vectora + mycounter, batchsize * sizeof(double), cudaMemcpyHostToDevice, stream1);
            __cudampi__memcpyAsync(devPtrb, vectorb + mycounter, batchsize * sizeof(double), cudaMemcpyHostToDevice, stream1);
            gettimeofday(&memcpy_end, NULL);
            time_memcpy += (memcpy_end.tv_sec - memcpy_start.tv_sec) + ((memcpy_end.tv_usec - memcpy_start.tv_usec)/1e6);

            // Timing kernel launch
            gettimeofday(&kernel_start, NULL);
            __cudampi__kernelInStream(devPtr, stream1);
            gettimeofday(&kernel_end, NULL);
            time_kernel += (kernel_end.tv_sec - kernel_start.tv_sec) + ((kernel_end.tv_usec - kernel_start.tv_usec)/1e6);

            // Timing memcpy operations
            gettimeofday(&memcpy_start, NULL);
            __cudampi__memcpyAsync(vectorc + mycounter, devPtrc, batchsize * sizeof(double), cudaMemcpyDeviceToHost, stream1);
            gettimeofday(&memcpy_end, NULL);
            time_memcpy += (memcpy_end.tv_sec - memcpy_start.tv_sec) + ((memcpy_end.tv_usec - memcpy_start.tv_usec)/1e6);

            if (streamcount == 2) 
            {
                // Timing the index fetching block for the second stream
                gettimeofday(&index_start, NULL);
                mycounter = __cudampi__getnextchunkindex(&globalcounter, batchsize, VECTORSIZE);
                counterrr += 1;
                gettimeofday(&index_end, NULL);
                time_index += (index_end.tv_sec - index_start.tv_sec) + ((index_end.tv_usec - index_start.tv_usec)/1e6);

                if (mycounter >= VECTORSIZE)
                {
                    finish = 1;
                } 
                else 
                {
                    // Timing memcpy operations
                    gettimeofday(&memcpy_start, NULL);
                    __cudampi__memcpyAsync(devPtra2, vectora + mycounter, batchsize * sizeof(double), cudaMemcpyHostToDevice, stream2);
                    __cudampi__memcpyAsync(devPtrb2, vectorb + mycounter, batchsize * sizeof(double), cudaMemcpyHostToDevice, stream2);
                    gettimeofday(&memcpy_end, NULL);
                    time_memcpy += (memcpy_end.tv_sec - memcpy_start.tv_sec) + ((memcpy_end.tv_usec - memcpy_start.tv_usec)/1e6);

                    // Timing kernel launch
                    gettimeofday(&kernel_start, NULL);
                    __cudampi__kernelInStream(devPtr2, stream2);
                    gettimeofday(&kernel_end, NULL);
                    time_kernel += (kernel_end.tv_sec - kernel_start.tv_sec) + ((kernel_end.tv_usec - kernel_start.tv_usec)/1e6);

                    // Timing memcpy operations
                    gettimeofday(&memcpy_start, NULL);
                    __cudampi__memcpyAsync(vectorc + mycounter, devPtrc2, batchsize * sizeof(double), cudaMemcpyDeviceToHost, stream2);
                    gettimeofday(&memcpy_end, NULL);
                    time_memcpy += (memcpy_end.tv_sec - memcpy_start.tv_sec) + ((memcpy_end.tv_usec - memcpy_start.tv_usec)/1e6);
                }
            }
        }

        privatecounter++;
        if (privatecounter % 30 == 0) 
        {
            // Timing the synchronization block
            gettimeofday(&sync_start, NULL);
            __cudampi__deviceSynchronize();
            gettimeofday(&sync_end, NULL);
            time_sync += (sync_end.tv_sec - sync_start.tv_sec) + ((sync_end.tv_usec - sync_start.tv_usec)/1e6);
        }

    } while (!finish);

    gettimeofday(&stop_p, NULL);
    double total_time = (double)(stop_p.tv_sec - start_p.tv_sec) + ((double)(stop_p.tv_usec - start_p.tv_usec) / 1e6);

    // Printing the measured times for each block
    log_message(LOG_INFO, "Thread %d: Allocation time=%f s, Index fetching time=%f s, Memcpy time=%f s, Kernel launch time=%f s, Synchronization time=%f s, Overall time=%f s, counter=%d\n", 
        omp_get_thread_num(), time_alloc, time_index, time_memcpy, time_kernel, time_sync, total_time, counterrr);

    __cudampi__deviceSynchronize();
    __cudampi__streamDestroy(stream1);
    if (streamcount == 2)
    {
        __cudampi__streamDestroy(stream2);
    }
}

  gettimeofday(&stop, NULL);
  log_message(LOG_INFO, "Main elapsed time=%f\n", (double)((stop.tv_sec - start.tv_sec) + (double)(stop.tv_usec - start.tv_usec) / 1000000.0));

  __cudampi__terminateMPI();

  gettimeofday(&stoptotal, NULL);
  log_message(LOG_INFO, "Total elapsed time=%f\n", (double)((stoptotal.tv_sec - starttotal.tv_sec) + (double)(stoptotal.tv_usec - starttotal.tv_usec) / 1000000.0));
}
