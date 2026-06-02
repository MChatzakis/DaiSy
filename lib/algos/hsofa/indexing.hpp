#ifndef SOFA_INDEXING_HPP
#define SOFA_INDEXING_HPP

#include <fftw3.h>
#include <pthread.h>

#include "../../isax/iSAXIndex.hpp"
#include "../../isax/iSAXPqueue.hpp"

namespace daisy
{
// structs that bring the data of each "worker"
    {
        isax_index        *index;
        float             *raw_database;
        unsigned long     *shared_start_number;
        unsigned long      stop_number;
        int                workernumber;
        int                total_workernumber;
        pthread_barrier_t *lock_barrier1;
        pthread_barrier_t *lock_barrier2;
        pthread_mutex_t   *lock_firstnode;
        int               *node_counter;
        int               *nodeid;
        int                read_block_length;
        float            **bins;
        float              norm_factor;
        bool               is_norm;
        float             *ts_buf;
        fftwf_complex     *ts_out;
        fftwf_plan         plan_forward;
        float             *fft_transform;
    };

    struct SOFA_bins_worker
    {
        isax_index    *index;
        float        **dft_mem_array;
        float         *raw_database;
        unsigned long  start_number;
        unsigned long  stop_number;
        long           records;
        long           records_offset;
        int            workernumber;
        float          norm_factor;
        bool           is_norm;
        int            coeff_number;
        float         *ts_buf;
        fftwf_complex *ts_out;
        fftwf_plan     plan_forward;
        float         *fft_transform;
    };

    struct SOFA_divide_worker_data
    {
        float        **dft_mem_array;
        float        **bins;
        unsigned long  start_number;
        unsigned long  stop_number;
        unsigned int   sample_size;
        int            num_symbols;
        int            histogram_type;
    };

    void sofa_fft_from_ts(
        int ts_length, float norm_factor, bool is_norm,
        fftwf_complex *ts_out, float *transform, fftwf_plan plan_forward,
        int coeff_number);

    void sofa_sfa_from_fft(
        float *transform, sax_type *sax_out, float **bins,
        int paa_segments, int cardinality);

    void *sofa_index_creation_worker(void *transferdata);

}

#endif
