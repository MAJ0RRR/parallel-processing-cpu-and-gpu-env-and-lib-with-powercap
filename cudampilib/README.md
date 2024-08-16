# CUDAMPILIB

## Getting started

```
cd existing_repo
git remote add origin https://kask.eti.pg.gda.pl/gitlab/pczarnul/cudampilib.git
git branch -M main
git push -uf origin main
```

## CUDAMPILIB

## Description

This project contains software that allows to develop parallel programs for a GPU cluster (several nodes, each of which contains 1+GPU),
either homogeneous or heterogeneous, using a model that is similar to the traditional approach for a single node shared memory system with 1+GPU i.e. OpenMP+CUDA where OpenMP threads are launched for management GPUs and CUDA used to access te GPUs.
In this project a programmer can follow a similar model for a GPU cluster where OpenMP threads are launched for management of GPUs, either local or remote ones, while extended CUDA API/API based on CUDA is used to access the GPUs, either local or remote.
Communication with remote GPUs is done with MPI. The implementation supports load balancing of data chunks among the GPUs and also overlapping Communication and computations using multiple CUDA streams.

The framework can optimize either:

1. execution time
2. execution time with an upper power limit for all GPUs selected for computations (set by the programmer - see the examples).

See the provided examples.

## Installation

Unpack the code, make sure that you have MPI, OpenMP and CUDA installed. Tested on Linux.

Source files can be compiled with:

./compile

## Usage

See the examples in scripts run* and modify accordingly.
Specifically, you might want to modify the MPI hostfile for your system. 

## Authors and acknowledgment

Copyright 2023 Paweł Czarnul pczarnul@eti.pg.edu.pl

## License

MIT License

Copyright 2023 Paweł Czarnul pczarnul@eti.pg.edu.pl

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to de
al in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sel
l copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITN
ESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY
, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTW
ARE.


## Project status

Active, open for cooperation.

## Files

appkernel*.cu - examples of simple kernels
app*.c - examples of actual applications (code of MPI's process 0), app-streams*.c are examples of applications using 2 CUDA streams per device
cudampislave.c - code of an MPI process other than the one with rank 0 i.e. ranks 1+ (running on remote nodes and acting as proxies to GPUs on remote nodes)
cudampi*.h/.c - code of the library
compile - compilation script
run* - scripts for running the code (adjust as necessary for your system/configuration)

