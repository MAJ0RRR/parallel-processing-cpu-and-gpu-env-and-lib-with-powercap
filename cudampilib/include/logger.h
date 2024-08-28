#ifndef LOGGER_H
#define LOGGER_H

#include <stdio.h>
#include <time.h>
#include <stdarg.h>

typedef enum {
    LOG_DEBUG,
    LOG_INFO,
    LOG_WARN,
    LOG_ERROR
} LogLevel;

static const char* LOG_LEVEL_NAMES[] = {
    "DEBUG", "INFO", "WARN", "ERROR"
};

#ifdef ENABLE_LOGGING
static void log_message(LogLevel level, const char *format, ...) {
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
}
#else
static void log_message(LogLevel level, const char *format, ...) {}
#endif // ENABLE_LOGGING

#endif // LOGGER_H
