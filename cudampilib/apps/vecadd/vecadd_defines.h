#ifndef VECADD_DEFINES_H
#define VECADD_DEFINES_H

#define VECADD_BATCH_SIZE 100000
#define VECADD_BLOCKS_IN_GRID 100
#define VECADD_THREADS_IN_BLOCK (VECADD_BATCH_SIZE / VECADD_BLOCKS_IN_GRID)
#define VECADD_VECTOR_SIZE 80000000

#endif // VECADD_DEFINES_H