#include "../hsofa/Sofa.hpp"

#ifdef SOFA_FFTW_ENABLED

#include <algorithm>
#include <cmath>
#include <cstring>

#include "../../isax/iSAXIndex.hpp"
#include "../../isax/iSAXSearch.hpp"

namespace daisy
{

    void sofa_fft_from_ts(
        int ts_length, float norm_factor, bool is_norm,
        fftwf_complex *ts_out, float *transform, fftwf_plan plan_forward,
        int coeff_number)
    {
        fftwf_execute(plan_forward);
        ts_out[0][1] = 0.0f;

        int j = 0;
        int start_offset = is_norm ? 1 : 0;
        for (int k = start_offset; k < coeff_number / 2 + start_offset; ++k, j += 2) {
            transform[j]     = ts_out[k][0];
            transform[j + 1] = ts_out[k][1] * -1.0f;
        }
        for (int i = 0; i < coeff_number; ++i)
            transform[i] *= norm_factor;
    }

    void sofa_sfa_from_fft(
        float *transform, sax_type *sax_out, float **bins,
        int paa_segments, int cardinality)
    {
        for (int k = 0; k < paa_segments; ++k) {
            unsigned int c = 0;
            for (; c < (unsigned int)(cardinality - 1); ++c)
                if (transform[k] < bins[k][c]) break;
            sax_out[k] = (sax_type)c;
        }
    }

    static void *sofa_bins_worker(void *transferdata)
    {
        SOFA_bins_worker *d = (SOFA_bins_worker *)transferdata;
        int ts_length       = d->index->settings->timeseries_size;

        for (long i = 0; i < d->records; ++i) {
            unsigned long db_idx = d->start_number + (unsigned long)i;
            if (db_idx >= d->stop_number) break;

            memcpy(d->ts_buf, &d->raw_database[db_idx * ts_length], sizeof(float) * ts_length);
            sofa_fft_from_ts(ts_length, d->norm_factor, d->is_norm,
                             d->ts_out, d->fft_transform, d->plan_forward, d->coeff_number);

            long slot = (long)d->workernumber * d->records_offset + i;
            for (int j = 0; j < d->coeff_number; ++j)
                d->dft_mem_array[j][slot] = d->fft_transform[j];
        }
        return nullptr;
    }

    static int sofa_compare_float(const void *a, const void *b)
    {
        float fa = *(const float *)a;
        float fb = *(const float *)b;
        return (fa < fb) ? -1 : (fa > fb) ? 1 : 0;
    }

    static void *sofa_divide_worker(void *transferdata)
    {
        SOFA_divide_worker_data *d = (SOFA_divide_worker_data *)transferdata;

        for (unsigned long j = d->start_number; j < d->stop_number; ++j)
            qsort(d->dft_mem_array[j], d->sample_size, sizeof(float), sofa_compare_float);

        for (unsigned long i = d->start_number; i < d->stop_number; ++i) {
            float *col = d->dft_mem_array[i];
            if (d->histogram_type == 1) {
                float depth     = (float)d->sample_size / (float)d->num_symbols;
                float bin_index = 0.0f;
                for (int j = 0; j < d->num_symbols - 1; ++j) {
                    bin_index += depth;
                    d->bins[i][j] = col[(int)bin_index];
                }
            } else {
                float first = col[0];
                float last  = col[d->sample_size - 1];
                float width = (last - first) / (float)d->num_symbols;
                for (int j = 0; j < d->num_symbols - 1; ++j)
                    d->bins[i][j] = width * (float)(j + 1) + first;
            }
        }
        return nullptr;
    }

    void *sofa_index_creation_worker(void *transferdata)
    {
        SOFA_index_worker  *d         = (SOFA_index_worker *)transferdata;
        isax_index         *index     = d->index;
        int                 ts_length = index->settings->timeseries_size;
        int                 paa_segs  = index->settings->paa_segments;

        sax_type           *sax = (sax_type *)malloc(sizeof(sax_type) * paa_segs);
        file_position_type *pos = (file_position_type *)malloc(sizeof(file_position_type));

        unsigned long roundfinish;
        while (1) {
            unsigned long start = __sync_fetch_and_add(d->shared_start_number,
                                                       (unsigned long)d->read_block_length);
            if (start > d->stop_number) break;
            roundfinish = (start > d->stop_number - (unsigned long)d->read_block_length)
                ? d->stop_number
                : std::min(d->stop_number, start + (unsigned long)d->read_block_length);

            for (unsigned long i = start; i < roundfinish; i++) {
                memcpy(d->ts_buf, &d->raw_database[i * ts_length], sizeof(float) * ts_length);
                sofa_fft_from_ts(ts_length, d->norm_factor, d->is_norm,
                                 d->ts_out, d->fft_transform, d->plan_forward, paa_segs);
                sofa_sfa_from_fft(d->fft_transform, sax, d->bins,
                                  paa_segs, index->settings->sax_alphabet_cardinality);

                *pos = (file_position_type)(i * ts_length);
                memcpy(&index->sax_cache[i * paa_segs], sax, sizeof(sax_type) * paa_segs);
                isax_pRecBuf_index_insert_inmemory(index, sax, pos,
                    d->lock_firstnode, d->workernumber, d->total_workernumber);
            }
        }

        free(sax);
        free(pos);

        pthread_barrier_wait(d->lock_barrier1);
        pthread_barrier_wait(d->lock_barrier2);

        isax_node_record *r = (isax_node_record *)malloc(sizeof(isax_node_record));
        while (1) {
            int j = __sync_fetch_and_add(d->node_counter, 1);
            if (j >= index->fbl->number_of_buffers) break;

            parallel_fbl_soft_buffer *cur =
                &((parallel_first_buffer_layer *)(index->fbl))->soft_buffers[j];
            if (!cur->initialized) continue;

            bool have_record = false;
            for (int k = 0; k < d->total_workernumber; k++) {
                if (cur->buffer_size[k] > 0) have_record = true;
                for (int ii = 0; ii < cur->buffer_size[k]; ii++) {
                    r->sax = (sax_type *)&(cur->sax_records[k][ii * paa_segs]);
                    r->position = (file_position_type *)
                                  &((file_position_type *)(cur->pos_records[k]))[ii];
                    r->insertion_mode = (insertion_mode)(NO_TMP | PARTIAL);
                    add_record_to_node(index, cur->node, r, 1);
                }
            }
            if (have_record)
                flush_subtree_leaf_buffers_inmemory(index, cur->node);
        }
        free(r);
        return nullptr;
    }

    void Sofa::sfaBinsInit()
    {
        bins = new float *[paa_segments];
        for (int i = 0; i < paa_segments; ++i) {
            bins[i] = new float[sax_cardinality - 1];
            for (int j = 0; j < sax_cardinality - 1; ++j)
                bins[i][j] = FLT_MAX;
        }
    }

    void Sofa::sfaFreeBins()
    {
        if (bins) {
            for (int i = 0; i < paa_segments; ++i) delete[] bins[i];
            delete[] bins;
            bins = nullptr;
        }
        delete[] coefficients;
        coefficients = nullptr;
    }

    void Sofa::sfaSetBins()
    {
        unsigned int actual_sample =
            (unsigned int)std::min((long)sample_size, (long)n_database);
        float norm_factor = std::sqrt(2.0f / (float)dim);

        float **dft_mem = new float *[paa_segments];
        for (int k = 0; k < paa_segments; ++k)
            dft_mem[k] = new float[actual_sample]();

        int  nw         = index_workers;
        long per_worker = actual_sample / nw;

        SOFA_bins_worker *bw   = new SOFA_bins_worker[nw];
        pthread_t        *tids = new pthread_t[nw];

        for (int i = 0; i < nw; i++) {
            bw[i].index          = index;
            bw[i].dft_mem_array  = dft_mem;
            bw[i].raw_database   = database;
            bw[i].start_number   = (unsigned long)i * per_worker;
            bw[i].stop_number    = (unsigned long)(i + 1) * per_worker;
            bw[i].records        = per_worker;
            bw[i].records_offset = per_worker;
            bw[i].workernumber   = i;
            bw[i].norm_factor    = norm_factor;
            bw[i].is_norm        = is_norm;
            bw[i].coeff_number   = paa_segments;

            bw[i].ts_buf        = (float *)fftwf_malloc(sizeof(float) * dim);
            bw[i].ts_out        = (fftwf_complex *)fftwf_malloc(
                                       sizeof(fftwf_complex) * (dim / 2 + 1));
            bw[i].plan_forward  = fftwf_plan_dft_r2c_1d(
                                       (int)dim, bw[i].ts_buf, bw[i].ts_out, FFTW_ESTIMATE);
            bw[i].fft_transform = (float *)fftwf_malloc(sizeof(float) * dim);
        }
        bw[nw - 1].records     = actual_sample - (long)(nw - 1) * per_worker;
        bw[nw - 1].stop_number = actual_sample;

        for (int i = 0; i < nw; i++)
            pthread_create(&tids[i], nullptr, sofa_bins_worker, &bw[i]);
        for (int i = 0; i < nw; i++)
            pthread_join(tids[i], nullptr);

        for (int i = 0; i < nw; i++) {
            fftwf_destroy_plan(bw[i].plan_forward);
            fftwf_free(bw[i].ts_buf);
            fftwf_free(bw[i].ts_out);
            fftwf_free(bw[i].fft_transform);
        }

        SOFA_divide_worker_data *dw      = new SOFA_divide_worker_data[nw];
        long                     segs_pw = paa_segments / nw;
        for (int i = 0; i < nw; i++) {
            dw[i].dft_mem_array  = dft_mem;
            dw[i].bins           = bins;
            dw[i].start_number   = (unsigned long)i * segs_pw;
            dw[i].stop_number    = (i < nw - 1)
                                 ? (unsigned long)(i + 1) * segs_pw
                                 : (unsigned long)paa_segments;
            dw[i].sample_size    = actual_sample;
            dw[i].num_symbols    = sax_cardinality;
            dw[i].histogram_type = histogram_type;
            pthread_create(&tids[i], nullptr, sofa_divide_worker, &dw[i]);
        }
        for (int i = 0; i < nw; i++)
            pthread_join(tids[i], nullptr);

        for (int k = 0; k < paa_segments; ++k) delete[] dft_mem[k];
        delete[] dft_mem;
        delete[] bw;
        delete[] dw;
        delete[] tids;

        fprintf(stderr, ">>> SOFA: Finished binning (sample=%u, %s)\n",
                actual_sample, histogram_type == 1 ? "equi-depth" : "equi-width");
    }

    void Sofa::buildIndex(DataSource *data_source)
    {
        this->dim        = data_source->getDim();
        this->n_database = data_source->getTotalRecords();

        if (this->n_database == 0) {
            data_source->reset();
            idx_t count = 0;
            float *dummy = new float[this->dim];
            while (data_source->nextRecord(dummy)) count++;
            delete[] dummy;
            this->n_database = count;
            data_source->reset();
        }

        data_source->reset();
        const float *raw = data_source->rawPointer();
        if (raw != nullptr) {
            this->database      = const_cast<float *>(raw);
            this->owns_database = false;
        } else {
            this->database = new float[this->n_database * this->dim];
            float *record  = new float[this->dim];
            idx_t idx = 0;
            while (data_source->nextRecord(record))
                std::copy(record, record + this->dim, this->database + idx++ * this->dim);
            delete[] record;
            this->owns_database = true;
        }

        // isax_index_settings_init expects sax_bit_cardinality (number of bits),
        // not the alphabet size. Convert: sax_cardinality = 8 symbols → 3 bits.
        int sax_bit_card = static_cast<int>(std::round(std::log2(static_cast<double>(this->sax_cardinality))));
        this->index_settings = isax_index_settings_init("",
            this->dim, this->paa_segments, sax_bit_card,
            this->leaf_size, this->min_leaf_size,
            this->initial_lbl_size, this->flush_limit,
            this->initial_fbl_size, this->total_loaded_leaves,
            this->tight_bound, 0, 1, 1);

        this->index = isax_index_init_inmemory(this->index_settings);
        isax_index *index = this->index;
        index->sax_file   = NULL;

        sfaBinsInit();
        sfaSetBins();

        float norm_factor       = std::sqrt(2.0f / (float)this->dim);
        int   read_block_length = this->read_block_length;
        unsigned long shared_start = 0;
        int node_counter = 0;

        index->sax_cache = (sax_type *)malloc(
            sizeof(sax_type) * index->settings->paa_segments * this->n_database);

        pthread_barrier_t lock_barrier1, lock_barrier2;
        pthread_barrier_init(&lock_barrier1, nullptr, this->index_workers + 1);
        pthread_barrier_init(&lock_barrier2, nullptr, this->index_workers + 1);
        pthread_mutex_t lock_firstnode = PTHREAD_MUTEX_INITIALIZER;

        destroy_fbl(index->fbl);
        index->fbl = (first_buffer_layer *)initialize_pRecBuf(
            index->settings->initial_fbl_buffer_size,
            (int)pow(2, index->settings->paa_segments),
            index->settings->max_total_buffer_size +
                DISK_BUFFER_SIZE * (PROGRESS_CALCULATE_THREAD_NUMBER - 1),
            index, this->index_workers);

        int *nodeid   = (int *)malloc(sizeof(int) * index->fbl->number_of_buffers);
        int *nodesize = (int *)malloc(sizeof(int) * index->fbl->number_of_buffers);

        pthread_t        *tids = new pthread_t[this->index_workers];
        SOFA_index_worker *iw  = new SOFA_index_worker[this->index_workers];

        for (int i = 0; i < this->index_workers; i++) {
            iw[i].index               = index;
            iw[i].raw_database        = this->database;
            iw[i].shared_start_number = &shared_start;
            iw[i].stop_number         = this->n_database;
            iw[i].workernumber        = i;
            iw[i].total_workernumber  = this->index_workers;
            iw[i].lock_barrier1       = &lock_barrier1;
            iw[i].lock_barrier2       = &lock_barrier2;
            iw[i].lock_firstnode      = &lock_firstnode;
            iw[i].node_counter        = &node_counter;
            iw[i].nodeid              = nodeid;
            iw[i].read_block_length   = read_block_length;
            iw[i].bins                = this->bins;
            iw[i].norm_factor         = norm_factor;
            iw[i].is_norm             = this->is_norm;

            iw[i].ts_buf        = (float *)fftwf_malloc(sizeof(float) * this->dim);
            iw[i].ts_out        = (fftwf_complex *)fftwf_malloc(
                                       sizeof(fftwf_complex) * (this->dim / 2 + 1));
            iw[i].plan_forward  = fftwf_plan_dft_r2c_1d(
                                       (int)this->dim, iw[i].ts_buf, iw[i].ts_out, FFTW_ESTIMATE);
            iw[i].fft_transform = (float *)fftwf_malloc(sizeof(float) * this->dim);

            pthread_create(&tids[i], nullptr, sofa_index_creation_worker, &iw[i]);
        }

        pthread_barrier_wait(&lock_barrier1);
        pthread_barrier_wait(&lock_barrier2);

        for (int i = 0; i < this->index_workers; i++)
            pthread_join(tids[i], nullptr);

        for (int i = 0; i < this->index_workers; i++) {
            fftwf_destroy_plan(iw[i].plan_forward);
            fftwf_free(iw[i].ts_buf);
            fftwf_free(iw[i].ts_out);
            fftwf_free(iw[i].fft_transform);
        }

        __sync_fetch_and_add(&index->total_records, this->n_database);
        index->sax_cache_size = index->total_records;
        fprintf(stderr, ">>> SOFA: Finished indexing\n");

        pthread_barrier_destroy(&lock_barrier1);
        pthread_barrier_destroy(&lock_barrier2);

        delete[] tids;
        delete[] iw;
        free(nodeid);
        free(nodesize);
    }

}

#endif
