#ifndef LOGGER_GPU_H
#define LOGGER_GPU_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

enum LogLevelGpu {
    LOG_DEBUG_GPU,
    LOG_WARN_GPU,
    LOG_INFO_GPU,
    LOG_ERROR_GPU
};

__device__ const char* LOG_LEVEL_NAMES_GPU[] = {
    "DEBUG", "WARN", "INFO", "ERROR"
};

#ifdef ENABLE_LOGGING_GPU
    #define LOG_LEVEL_THRESHOLD_GPU LOG_WARN_GPU   // All logs enabled expect DEBUG
#endif

#ifdef ENABLE_LOGGING_GPU // All logs expect DEBUG in stdout
    __device__ void log_message_gpu(LogLevelGpu level, const char *message) {
        if (level >= LOG_LEVEL_THRESHOLD_GPU)
        {
            printf("[%s] [Block: %d, Thread: %d] %s\n", 
                LOG_LEVEL_NAMES_GPU[level], 
                blockIdx.x, threadIdx.x, message);
        }  
    }    
#else
    #ifdef ENABLE_LOGGING_GPU_DEBUG // All logs in user defined file, buffer needs to be provided from GPU kernel
        void log_message_gpu(const char* filename, char *logBuffer, size_t bufferSize, LogLevel level) {
            FILE *log_file = fopen(filename, "a");
            if (log_file == NULL) {
                fprintf(stderr, "Error opening log file!\n");
                return;
            }

            time_t now;
            time(&now);
            struct tm *local = localtime(&now);

            char time_str[20];
            strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", local);

            fprintf(log_file, "[%s] [%s]\n", time_str, LOG_LEVEL_NAMES[level]);
            for (size_t i = 0; i < bufferSize; ++i) {
                fprintf(log_file, "%s", &logBuffer[i * 128]);
            }
            
            fclose(log_file);
            }
    #else // Logs disabled
        __device__ void log_message_gpu(LogLevelGpu level, const char *message) {}
    #endif // ENABLE_LOGGING_GPU_FILE
#endif // ENABLE_LOGGING_GPU
#endif // LOGGER_GPU_H