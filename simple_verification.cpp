#include "commons/dataloaders.hpp"
#include "lib/algos/Bruteforce.hpp"
#include "lib/algos/LbBruteforce.hpp"
#include <iostream>
#include <chrono>
#include <cmath>

// Helper function to compare floating point numbers with tolerance
bool isClose(float a, float b, float tolerance = 1e-4) {
    return std::abs(a - b) < tolerance;
}

int main() {
    printf("=== L2 Squared Distance Algorithm Verification ===\n\n");
    
    // Test configuration - smaller for quick testing
    diNoLib::idx_t n_database = 1000;
    unsigned long long dim = 16;
    unsigned long long n_query = 5;
    diNoLib::idx_t k = 3;
    
    printf("Configuration:\n");
    printf("  Database size: %llu\n", n_database);
    printf("  Dimension: %llu\n", dim);
    printf("  Queries: %llu\n", n_query);
    printf("  k: %llu\n\n", k);
    
    // Generate test data with fixed seed for reproducibility
    float *database = loadRandomData(n_database, dim, true, 123);
    float *query = loadRandomData(n_query, dim, true, 456);
    
    // Results storage
    diNoLib::idx_t *I_bf = new diNoLib::idx_t[n_query * k];
    float *D_bf = new float[n_query * k];
    diNoLib::idx_t *I_lbbf = new diNoLib::idx_t[n_query * k];
    float *D_lbbf = new float[n_query * k];
    
    // Test 1: Bruteforce
    printf("1. Testing Bruteforce Algorithm:\n");
    diNoLib::BruteForceSearch bf_search(diNoLib::DistanceType::L2_SQUARED);
    
    auto start = std::chrono::high_resolution_clock::now();
    bf_search.buildIndex(database, n_database, dim);
    auto build_time = std::chrono::high_resolution_clock::now();
    
    bf_search.searchIndex(query, n_query, k, I_bf, D_bf);
    auto search_time = std::chrono::high_resolution_clock::now();
    
    auto bf_build_ms = std::chrono::duration_cast<std::chrono::microseconds>(build_time - start).count();
    auto bf_search_ms = std::chrono::duration_cast<std::chrono::microseconds>(search_time - build_time).count();
    
    printf("  Build time: %ld μs\n", bf_build_ms);
    printf("  Search time: %ld μs\n", bf_search_ms);
    
    // Test 2: LbBruteforce  
    printf("\n2. Testing LbBruteforce Algorithm:\n");
    diNoLib::LbBruteforce lbbf_search(diNoLib::DistanceType::L2_SQUARED);
    
    start = std::chrono::high_resolution_clock::now();
    lbbf_search.buildIndex(database, n_database, dim);
    build_time = std::chrono::high_resolution_clock::now();
    
    lbbf_search.searchIndex(query, n_query, k, I_lbbf, D_lbbf);
    search_time = std::chrono::high_resolution_clock::now();
    
    auto lbbf_build_ms = std::chrono::duration_cast<std::chrono::microseconds>(build_time - start).count();
    auto lbbf_search_ms = std::chrono::duration_cast<std::chrono::microseconds>(search_time - build_time).count();
    
    printf("  Build time: %ld μs\n", lbbf_build_ms);
    printf("  Search time: %ld μs\n", lbbf_search_ms);
    
    // Test 3: Compare Results
    printf("\n3. Comparing Results:\n");
    bool results_match = true;
    int mismatches = 0;
    
    for (diNoLib::idx_t qi = 0; qi < n_query; qi++) {
        printf("Query %llu: ", qi);
        for (diNoLib::idx_t j = 0; j < k; j++) {
            diNoLib::idx_t idx1 = I_bf[qi * k + j];
            diNoLib::idx_t idx2 = I_lbbf[qi * k + j];
            float dist1 = D_bf[qi * k + j];
            float dist2 = D_lbbf[qi * k + j];
            
            if (idx1 != idx2 || !isClose(dist1, dist2)) {
                printf("❌[%llu,%llu: BF=%llu/%.4f vs LbBF=%llu/%.4f] ", 
                       j, j, idx1, dist1, idx2, dist2);
                results_match = false;
                mismatches++;
            } else {
                printf("✅[%llu] ", j);
            }
        }
        printf("\n");
    }
    
    printf("\n=== VERIFICATION SUMMARY ===\n");
    if (results_match) {
        printf("✅ SUCCESS: Both algorithms produce IDENTICAL results!\n");
        printf("✅ L2 Squared distance implementation is CORRECT\n");
        
        // Performance comparison
        printf("\nPerformance Summary:\n");
        printf("  Bruteforce: Build=%ld μs, Search=%ld μs\n", bf_build_ms, bf_search_ms);
        printf("  LbBruteforce: Build=%ld μs, Search=%ld μs\n", lbbf_build_ms, lbbf_search_ms);
        
        long total_bf = bf_build_ms + bf_search_ms;
        long total_lbbf = lbbf_build_ms + lbbf_search_ms;
        
        if (total_lbbf < total_bf) {
            printf("  🚀 LbBruteforce is %.2fx faster overall!\n", 
                   (float)total_bf / total_lbbf);
        } else if (total_bf < total_lbbf) {
            printf("  ⚠️  Bruteforce is %.2fx faster (lower bound overhead on small data)\n", 
                   (float)total_lbbf / total_bf);
        }
    } else {
        printf("❌ FAILURE: Algorithms produce DIFFERENT results!\n");
        printf("❌ Found %d mismatches out of %llu total comparisons\n", mismatches, n_query * k);
        printf("❌ There may be a bug in one of the implementations\n");
    }
    
    // Cleanup
    delete[] database;
    delete[] query;
    delete[] I_bf;
    delete[] D_bf;
    delete[] I_lbbf;
    delete[] D_lbbf;
    
    printf("\n=== VERIFICATION COMPLETE ===\n");
    return results_match ? 0 : 1;
}
