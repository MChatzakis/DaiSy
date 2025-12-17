#include "Sing.hpp"

namespace diNoLib
{

    Sing::Sing(DistanceType distance_type)
        : SimilaritySearchAlgorithm(distance_type)
    {
    }

    void Sing::setNumThreads(int num_threads)
    {
        int max_threads = omp_get_max_threads();

        if (num_threads > max_threads) 
        {
            std::cerr << "[Warning] " << num_threads 
                    << " threads exceeds max available " << max_threads << " Using the max threads available.\n";
            this->num_threads = max_threads;
        } 
        else if (num_threads < 1) 
        {
            std::cerr << "[Warning] Thread count must be >= 1. Using 1.\n";
            this->num_threads = 1;
        } 
        else 
        {
            this->num_threads = num_threads;
        }
    } 

    int Sing::getNumThreads() const
    {
        return this->num_threads;
    } 

    void Sing::buildIndex(DataSource *data_source)
    {
        this->dim = data_source->getDim();
        this->n_database = data_source->getTotalRecords();

        // If n_database is 0, we need to determine it first
        if (this->n_database == 0)
        {
            // Reset and count
            data_source->reset();
            idx_t count = 0;
            float *dummy = new float[this->dim];
            while (data_source->nextRecord(dummy))
            {
                count++;
            }
            delete[] dummy;
            this->n_database = count;
            data_source->reset();
        }

        // Allocate and load all data
        this->database = new float[this->n_database * this->dim];
        float *record = new float[this->dim];
        idx_t idx = 0;
        while (data_source->nextRecord(record))
        {
            std::copy(record, record + this->dim, this->database + idx * this->dim);
            idx++;
        }
        delete[] record;
    }
    
    void Sing::searchIndex(const float *query, const idx_t n_query, const idx_t k, idx_t *I, float *D)
    {
        #pragma omp parallel num_threads(num_threads)
        {
            #pragma omp for 
            for (idx_t qi = 0; qi < n_query; qi++)
            {   
                std::priority_queue<std::pair<float, idx_t>> pq;
                const float *q_vec = query + qi * this->dim;

                float bound = FLT_MAX;  // initialize bound to max float

                for (idx_t dbi = 0; dbi < this->n_database; ++dbi)
                {
                    const float *db_vec = this->database + dbi * this->dim;
                    float dist = this->distance_computer->compute_dist(const_cast<float *>(q_vec), 
                                                                        const_cast<float *>(db_vec), 
                                                                        this->dim, 
                                                                        bound);
                    if ((idx_t)pq.size() < k) // maintain max-heap
                    {
                        pq.emplace(dist, dbi); // equivalent to pq.push(make_pair(dist, dbi));
                    }
                    else if (dist < pq.top().first) 
                    {
                        pq.pop();
                        pq.emplace(dist, dbi);                         
                        bound = pq.top().first; // update the bound variable for pruning
                    }

                }
                
                // store top-k results in reverse order
                for (idx_t j = k; j > 0; --j)
                {
                    D[qi * k + (j - 1)] = pq.top().first;
                    I[qi * k + (j - 1)] = pq.top().second;
                    pq.pop();
                }               
            }
        }
    }

    Sing::~Sing()
    {
        delete[] database;
    }
}