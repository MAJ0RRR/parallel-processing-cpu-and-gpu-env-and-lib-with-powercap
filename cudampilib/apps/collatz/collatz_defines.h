#ifndef COLLATZ_DEFINES_H
#define COLLATZ_DEFINES_H

#define COLLATZ_VECTORSIZE 200000000
#define COLLATZ_BATCH_SIZE 50000
#define COLLATZ_BLOCKS_IN_GRID 100
#define COLLATZ_THREADS_IN_BLOCK (COLLATZ_BATCH_SIZE / COLLATZ_BLOCKS_IN_GRID)

#endif // COLLATZ_DEFINES_H