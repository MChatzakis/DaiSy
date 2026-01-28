#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <unordered_map>
#include <string>
#include <vector>

namespace dinoLib {

struct timer_data_t {
    double time;
    timeval start;
};

struct timer_manager_t {
    std::unordered_map<std::string, timer_data_t> *timer_map;
};

using TimerManager = timer_manager_t;

/* Init and Destroy global manager */
void timer_init(timer_manager_t *timer_manager);
void timer_destroy(timer_manager_t *timer_manager);

/* Timing and Benchmarking management*/
void timer_log(timer_manager_t *timer_manager, FILE *stream);
void timer_to_csv_file(timer_manager_t *timer_manager, const char *filename, bool include_header, bool append);
void timer_add(timer_manager_t *timer_manager, const char *timer_name);
void timer_remove(timer_manager_t *timer_manager, const char *timer_name);
void timer_start(timer_manager_t *timer_manager, const char *timer_name);
void timer_start_force(timer_manager_t *timer_manager, const char *timer_name);
void timer_stop(timer_manager_t *timer_manager, const char *timer_name);
void timer_set(timer_manager_t *timer_manager, const char *timer_name, const double val);
void timer_print(timer_manager_t *timer_manager, const char *timer_name, FILE *stream);

double timer_get(timer_manager_t *timer_manager, const char *timer_name);
char **timer_get_names(timer_manager_t *timer_manager);

double *timer_to_array(timer_manager_t *timer_manager, int *size);

} // namespace dinoLib