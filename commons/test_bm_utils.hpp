#ifndef TEST_BM_UTILS_HPP
#define TEST_BM_UTILS_HPP

#include <string>
#include "../lib/algos/SimilaritySearchAlgorithm.hpp"

/**
 * @brief Integer equality.
 *
 * @param a First index
 * @param b Second index
 * @return True if equal, false otherwise
 */
bool isclose(daisy::idx_t a, daisy::idx_t b);

/**
 * @brief Floating point ; A equivalent function of numpy.isclose -- absolute(a - b) <= (atol + rtol * absolute(b))
 *
 * @param a First value
 * @param b Second value
 * @param rtol Relative tolerance
 * @param atol Absolute tolerance
 * @return True if values are close, false otherwise
 */
bool isclose(double a, double b, double rtol = 1e-5, double atol = 1e-8);

/**
 * @brief Read a text file containing floating-point numbers and return them as a dynamically allocated array.
 *
 * @param filepath Path to the file
 * @param outSize Output variable storing the number of floats read
 * @return Pointer to the array of floats, or nullptr on error
 */
float *readFile(const std::string &filepath, size_t &outSize);

/**
 * @brief Extract filename from full path.
 *
 * @param path Full file path
 * @return Filename portion of the path
 */
std::string pathToFilename(std::string path);

/**
 * @brief Parse dataset filename to extract configuration parameters.
 *
 * @param filename Name of the dataset file
 * @param dim Output dimension of the data vectors
 * @param n_database Output number of vectors in the database
 * @param n_query Output number of query vectors
 * @param k Output number of nearest neighbors
 * @return True if all required parameters were successfully parsed, false otherwise
 */
bool parseFilenameForConfig(const std::string &filename,
                            const std::string &prefix,
                            daisy::idx_t &dim,
                            daisy::idx_t &n_database,
                            daisy::idx_t &n_query,
                            daisy::idx_t &k);

/**
 * @brief Compare brute-force search results with ground truth and report mismatches or close results.
 *
 * @param pathI Path to ground-truth index file
 * @param pathD Path to ground-truth distance file
 * @param I Computed index results
 * @param D Computed distance results
 * @param n_query Number of queries
 * @param k Number of top results
 */
void compareWithGroundTruth(const std::string &pathI,
                            const std::string &pathD,
                            const daisy::idx_t *I,
                            float *D,
                            daisy::idx_t n_query,
                            daisy::idx_t k,
                            double rtol = 1e-2,
                            double atol = 1e-8);
/**
 * @brief Assert that two size_t values are equal; if not, print an error message and terminate the program.
 *
 * @param a First value to compare
 * @param b Second value to compare
 * @param msg Message to display if the assertion fails
 */
void assert_eq(size_t a, size_t b, const std::string &msg);

/**
 * @brief Report a failure by printing an error message but continue execution.
 *
 * @param msg Message describing the failure
 */
void add_failure(const std::string &msg);

/**
 * @brief Configuration structure for similarity search tests
 */
struct SSTestConfig
{
    std::string name;
    std::string dataset_path;
    std::string query_path;
    std::string gt_I_prefix;
    std::string gt_D_prefix;
    int thread_count;
    int k_value;
};

#endif // TEST_BM_UTILS_HPP