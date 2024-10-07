#ifndef LOGGER_H
#define LOGGER_H

#include <stdio.h>
#include <time.h>
#include <stdarg.h>

typedef enum {
    LOG_DEBUG,
    LOG_WARN,
    LOG_INFO,
    LOG_ERROR
} LogLevel;

static const char* LOG_LEVEL_NAMES[] = {
    "DEBUG", "WARN", "INFO", "ERROR"
};

#ifdef ENABLE_LOGGING
    #define LOG_LEVEL_THRESHOLD LOG_WARN   // All logs enabled expect DEBUG
#endif

#ifdef ENABLE_LOGGING // All logs expect DEBUG in stdout
    static void log_message(LogLevel level, const char *format, ...) {
        if (level >= LOG_LEVEL_THRESHOLD)
        {
            time_t now;
            time(&now);
            struct tm *local = localtime(&now);
            
            char time_str[20];
            strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", local);
            
            fprintf(stderr, "[%s] [%s] ", time_str, LOG_LEVEL_NAMES[level]);
            
            va_list args;
            va_start(args, format);
            vfprintf(stderr, format, args);
            va_end(args);
            
            fprintf(stderr, "\n");
            fflush(stderr);
        }
    }
#else
    #ifdef ENABLE_LOGGING_DEBUG // All logs in logfile.txt
        static void log_message(LogLevel level, const char *format, ...) {
            FILE *log_file = fopen("logfile.txt", "a");
            if (log_file == NULL) {
                fprintf(stderr, "Error opening log file!\n");
                return;
            }
            
            time_t now;
            time(&now);
            struct tm *local = localtime(&now);
            
            char time_str[20];
            strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", local);
            
            fprintf(log_file, "[%s] [%s] ", time_str, LOG_LEVEL_NAMES[level]);
            
            va_list args;
            va_start(args, format);
            vfprintf(log_file, format, args);
            va_end(args);
            
            fprintf(log_file, "\n");
            
            fclose(log_file);
        }
    #else // Logs disabled
        static void log_message(LogLevel level, const char *format, ...) {}
    #endif // ENABLE_LOGGING_DEBUG
#endif // ENABLE_LOGGING
#endif // LOGGER_H
