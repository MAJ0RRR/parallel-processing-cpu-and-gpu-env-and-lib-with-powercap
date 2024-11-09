#ifndef UTILITY_H
#define UTILITY_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef ENABLE_OUTPUT_LOGS
void save_vector_output_double(double* start, int batchsize, const char* filename, const char* header) {
    struct stat st = {0};
    if (stat("logs", &st) == -1) {
        mkdir("logs", 0700);
    }

    char fullpath[256];
    snprintf(fullpath, sizeof(fullpath), "logs/%s", filename);

    FILE *file = fopen(fullpath, "w");
    fprintf(file, "%s\n", header);
     fprintf(file, "Array of size %d:\n", batchsize);
    for (int i = 0; i < batchsize; i++) {
        fprintf(file, "%f\n", start[i]);
    }
    fclose(file);
}

void save_vector_output_char(char* start, int batchsize, const char* filename, const char* header) {
    struct stat st = {0};
    if (stat("logs", &st) == -1) {
        mkdir("logs", 0700);
    }

    char fullpath[256];
    snprintf(fullpath, sizeof(fullpath), "logs/%s", filename);

    FILE *file = fopen(fullpath, "w");
    fprintf(file, "%s\n", header);
     fprintf(file, "Array of size %d:\n", batchsize);
    for (int i = 0; i < batchsize; i++) {
        fprintf(file, "%c\n", start[i]);
    }
    fclose(file);
}
#else
void save_vector_output_double(double* start, int batchsize, const char* filename, const char* header) { }
void save_vector_output_char(char* start, int batchsize, const char* filename, const char* header) { }
#endif

#endif // UTILITY_H