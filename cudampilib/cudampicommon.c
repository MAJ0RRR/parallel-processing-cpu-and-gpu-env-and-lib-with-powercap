/*
Copyright 2023 Paweł Czarnul pczarnul@eti.pg.edu.pl

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/
#include "cudampicommon.h"

float computeDevPerformance(struct timeval period) {
  // period is just the time between two events so compute performance as an inverse

  return 1000000.0 / (period.tv_sec * 1000000 + period.tv_usec);
}

float getGPUpower(int gpuid) {

  char buffer[500];
  char filename[100];
  float power;

  sprintf(filename, "/tmp/__cudampi__gpu_power.%d", gpuid);
  sprintf(buffer, "nvidia-smi -q -i %d -d POWER | grep \"Power Draw\" | tr -s ' ' | cut -d ' ' -f 5 > %s", gpuid, filename);

  system(buffer);

  FILE *fp = fopen(filename, "r");
  fscanf(fp, "%f", &power);
  fclose(fp);

  return power;
}

cudaError_t __cudampi__getCpuFreeThreads(int* count)
{
  // TODO
  return 0;
}

 cudaError_t getCpuEnergyUsed(float* lastEnergyMeasured, float* energyUsed) {
  // compute energy used from last energy measurement and update the variable

  FILE *file;
  unsigned long long energy_uj;
  float energy_joules;

  file = fopen("/sys/class/powercap/intel-rapl:0/energy_uj", "r");
  if (file == NULL) {
      perror("Failed to open energy_uj file");
      return cudaErrorUnknown ;
  }

  if (fscanf(file, "%llu", &energy_uj) != 1) {
      perror("Failed to read energy value");
      fclose(file);
      return cudaErrorUnknown ;
  }

  fclose(file);

  energy_joules = (float)energy_uj / 1e6;

  *energyUsed = energy_joules - lastEnergyMeasured;

  *lastEnergyMeasured = energy_joules;

  return cudaSuccess;
}
