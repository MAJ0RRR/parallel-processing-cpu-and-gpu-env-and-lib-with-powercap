/*
Copyright 2023 Paweł Czarnul pczarnul@eti.pg.edu.pl

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "cudampilib.h"
#include "apppatternlength.h"

#include <sys/time.h>



long long VECTORSIZE=400000000; // 800000000

char *vectora;
char *vectorc;

int batchsize=50000; //100000;

long long globalcounter=0; // access to it controlled within a critical section


int streamcount=1;
float powerlimit;


int main(int argc,char **argv) {

   struct timeval start,stop;
   struct timeval starttotal,stoptotal;

 
   gettimeofday(&starttotal,NULL);

    
  __cudampi__initializeMPI(argc,argv);

  int cudadevicescount=1;

  if (argc>1) streamcount=atoi(argv[1]);

  if (argc>2) { 
	  powerlimit=atof(argv[2]);
	  printf("\nSetting power limit=%f\n",powerlimit);
	fflush(stdout);
	__cudampi__setglobalpowerlimit(powerlimit);
  }


  __cudampi__getCUDAdevicescount(&cudadevicescount);


  cudaHostAlloc((void**)&vectora,sizeof(char)*(VECTORSIZE+PATTERNLENGTH),cudaHostAllocDefault);
  if (!vectora) {
    printf("\nNot enough memory.");
    exit(0);
  }

  cudaHostAlloc((void**)&vectorc,sizeof(char)*VECTORSIZE,cudaHostAllocDefault);
  if (!vectorc) {
    printf("\nNot enough memory.");
    exit(0);
  }


  for(long long i=0;i<(VECTORSIZE+PATTERNLENGTH);i++)
    vectora[i]=(1+i)%3;
  
  gettimeofday(&start,NULL);
  
  #pragma omp parallel num_threads(cudadevicescount)
  {


    long long mycounter;
    int finish=0;
    void *devPtra,*devPtrc;
    void *devPtra2,*devPtrc2;
  int i;
  cudaStream_t stream1;
  cudaStream_t stream2;
  int mythreadid=omp_get_thread_num();
  void *devPtr;
  void *devPtr2;
  long long privatecounter=0;



  
  __cudampi__cudaSetDevice(mythreadid);
#pragma omp barrier  
  __cudampi__cudaMalloc(&devPtra,(batchsize+PATTERNLENGTH)*sizeof(char));
  __cudampi__cudaMalloc(&devPtrc,batchsize*sizeof(char));

  __cudampi__cudaMalloc(&devPtr,2*sizeof(void *));


  __cudampi__cudaMalloc(&devPtra2,(batchsize+PATTERNLENGTH)*sizeof(char));
  __cudampi__cudaMalloc(&devPtrc2,batchsize*sizeof(char));

  __cudampi__cudaMalloc(&devPtr2,2*sizeof(void *));



  __cudampi__cudaStreamCreate(&stream1);

  __cudampi__cudaStreamCreate(&stream2);


  
  __cudampi__cudaMemcpyAsync(devPtr,&devPtra,sizeof(void *),cudaMemcpyHostToDevice,stream1);
  __cudampi__cudaMemcpyAsync(devPtr+sizeof(void *),&devPtrc,sizeof(void *),cudaMemcpyHostToDevice,stream1);

  __cudampi__cudaMemcpyAsync(devPtr2,&devPtra2,sizeof(void *),cudaMemcpyHostToDevice,stream2);
  __cudampi__cudaMemcpyAsync(devPtr2+sizeof(void *),&devPtrc2,sizeof(void *),cudaMemcpyHostToDevice,stream2);



  do {

  mycounter=__cudampi__getnextchunkindex(&globalcounter,batchsize,VECTORSIZE);
   

    if (mycounter>=VECTORSIZE) finish=1;
    else {
      
      __cudampi__cudaMemcpyAsync(devPtra,vectora+mycounter,(batchsize+PATTERNLENGTH)*sizeof(char),cudaMemcpyHostToDevice,stream1);
      
      
      
      __cudampi__kernelinstream(devPtr,stream1);
    
    
      __cudampi__cudaMemcpyAsync(vectorc+mycounter,devPtrc,batchsize*sizeof(char),cudaMemcpyDeviceToHost,stream1);
    
    }
    // do it again in the second stream
    if (streamcount==2)
    if (!finish) {
      

   mycounter=__cudampi__getnextchunkindex(&globalcounter,batchsize,VECTORSIZE);
    

    if (mycounter>=VECTORSIZE) finish=1;
    else {
      
      __cudampi__cudaMemcpyAsync(devPtra2,vectora+mycounter,(batchsize+PATTERNLENGTH)*sizeof(char),cudaMemcpyHostToDevice,stream2);
      
      
      
    __cudampi__kernelinstream(devPtr2,stream2);
    
    
    __cudampi__cudaMemcpyAsync(vectorc+mycounter,devPtrc2,batchsize*sizeof(char),cudaMemcpyDeviceToHost,stream2);
    
    }

      

  }
        
    privatecounter++;
    if (privatecounter%2) {
	    __cudampi__cudaDeviceSynchronize();

    }
    
  } while (!finish);

  
  __cudampi__cudaDeviceSynchronize();
  __cudampi__cudaStreamDestroy (stream1);  
  

  }

  gettimeofday(&stop,NULL);

  
  printf("Main elapsed time=%f\n",(double)((stop.tv_sec-start.tv_sec)+(double)(stop.tv_usec-start.tv_usec)/1000000.0));
  fflush(stdout);

  

  __cudampi__terminateMPI();  
  
  gettimeofday(&stoptotal,NULL);
  printf("Total elapsed time=%f\n",(double)((stoptotal.tv_sec-starttotal.tv_sec)+(double)(stoptotal.tv_usec-starttotal.tv_usec)/1000000.0));
  fflush(stdout);


}
