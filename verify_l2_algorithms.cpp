#include "commons/dataloaders.hpp"
#include "lib/algos/Bruteforce.hpp"
#include "lib/algos/LbBruteforce.hpp"
#include <cmath>
#include <chrono>
#include <iomanip>

// Helper function to compare floating point numbers with tolerance
bool isClose(float a, float b, float tolerance = 1e-5) {
    return std::abs(a - b) < tolerance;
}

// Helper function to compare result arrays
bool compareResults(const diNoLib::idx_t* I1, const float* D1, 
                    const diNoLib::idx_t* I2, const float* D2,
                    diNoLib::idx_t n_query, diNoLib::idx_t k) {
    bool identical = true;
    
    for (diNoLib::idx_t qi = 0; qi < n_query; qi++) {
        printf("Query %llu comparison:\n", qi);
        for (diNoLib::idx_t j = 0; j < k; j++) {
            diNoLib::idx_t idx1 = I1[qi * k + j];
            diNoLib::idx_t idx2 = I2[qi * k + j];
            float dist1 = D1[qi * k + j];
            float dist2 = D2[qi * k + j];
            
            printf("  %llu: BF[%llu, %.6f] vs LbBF[%llu, %.6f]", 
                   j, idx1, dist1, idx2, dist2);
            
            if (idx1 != idx2 || !isClose(dist1, dist2)) {
                printf(" ❌ MISMATCH!");
                identical = false;
            } else {
                printf(" ✅");
            }
            printf("\n");
        }
    }
    
    return identical;
}

int main() {
    printf("=== L2 Squared Distance Algorithm Verification ===\n\n");
    
    // Test configuration
    diNoLib::idx_t n_database = 10000;  // Smaller for faster testing
    unsigned long long dim = 32;        // Smaller dimension
    unsigned long long n_query = 5;
    diNoLib::idx_t k = 3;
    
    printf("Configuration:\n");
    printf("  Database size: %llu\n", n_database);
    printf("  Dimension: %llu\n", dim);
    printf("  Queries: %llu\n", n_query);
    printf("  k: %llu\n\n", k);
    
    // Generate test data
    printf("Generating test data...\n");
    float *database = loadRandomData(n_database, dim, true, 123);  // Fixed seed for reproducibility
    float *query = loadRandomData(n_query, dim, true, 456);        // Fixed seed for reproducibility
    
    // Results storage
    diNoLib::idx_t *I_bf = new diNoLib::idx_t[n_query * k];
    float *D_bf = new float[n_query * k];
    diNoLib::idx_t *I_lbbf = new diNoLib::idx_t[n_query * k];
    float *D_lbbf = new float[n_query * k];
    
    // Test 1: Bruteforce
    printf("\n1. Testing Bruteforce Algorithm:\n");
    diNoLib::BruteForceSearch bf_search(diNoLib::DistanceType::L2_SQUARED);
    
    auto start = std::chrono::high_resolution_clock::now();
    bf_search.buildIndex(database, n_database, dim);
    auto build_time = std::chrono::high_resolution_clock::now();
    
    bf_search.searchIndex(query, n_query, k, I_bf, D_bf);
    auto search_time = std::chrono::high_resolution_clock::now();
    
    auto bf_build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(build_time - start).count();
    auto bf_search_ms = std::chrono::duration_cast<std::chrono::milliseconds>(search_time - build_time).count();
    
    printf("  Build time: %ld ms\n", bf_build_ms);
    printf("  Search time: %ld ms\n", bf_search_ms);
    
    // Test 2: LbBruteforce  
    printf("\n2. Testing LbBruteforce Algorithm:\n");
    diNoLib::LbBruteforce lbbf_search(diNoLib::DistanceType::L2_SQUARED);
    
    start = std::chrono::high_resolution_clock::now();
    lbbf_search.buildIndex(database, n_database, dim);
    build_time = std::chrono::high_resolution_clock::now();
    
    lbbf_search.searchIndex(query, n_query, k, I_lbbf, D_lbbf);
    search_time = std::chrono::high_resolution_clock::now();
    
    auto lbbf_build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(build_time - start).count();
    auto lbbf_search_ms = std::chrono::duration_cast<std::chrono::milliseconds>(search_time - build_time).count();
    
    printf("  Build time: %ld ms\n", lbbf_build_ms);
    printf("  Search time: %ld ms\n", lbbf_search_ms);
    
    // Test 3: Compare Results
    printf("\n3. Comparing Results:\n");
    bool results_match = compareResults(I_bf, D_bf, I_lbbf, D_lbbf, n_query, k);
    
    printf("\n=== VERIFICATION SUMMARY ===\n");
    if (results_match) {
        printf("✅ SUCCESS: Both algorithms produce IDENTICAL results!\n");
        printf("✅ L2 Squared distance implementation is CORRECT\n");
        
        // Performance comparison
        printf("\nPerformance Summary:\n");
        printf("  Bruteforce: Build=%ld ms, Search=%ld ms, Total=%ld ms\n", 
               bf_build_ms, bf_search_ms, bf_build_ms + bf_search_ms);
        printf("  LbBruteforce: Build=%ld ms, Search=%ld ms, Total=%ld ms\n", 
               lbbf_build_ms, lbbf_search_ms, lbbf_build_ms + lbbf_search_ms);
        
        if (lbbf_search_ms < bf_search_ms) {
            printf("  🚀 LbBruteforce is %.1fx faster in search!\n", 
                   (float)bf_search_ms / lbbf_search_ms);
        }
    } else {
        printf("❌ FAILURE: Algorithms produce DIFFERENT results!\n");
        printf("❌ There may be a bug in one of the implementations\n");
    }
    
    // Test 4: Edge Cases
    printf("\n4. Testing Edge Cases:\n");
    
    // Test k = 1
    printf("  Testing k=1...");
    diNoLib::idx_t *I_edge = new diNoLib::idx_t[n_query * 1];
    float *D_edge = new float[n_query * 1];
    bf_search.searchIndex(query, n_query, 1, I_edge, D_edge);
    lbbf_search.searchIndex(query, n_query, 1, I_lbbf, D_lbbf);
    
    bool edge_match = true;
    for (diNoLib::idx_t qi = 0; qi < n_query; qi++) {
        if (I_edge[qi] != I_lbbf[qi] || !isClose(D_edge[qi], D_lbbf[qi])) {
            edge_match = false;
            break;
        }
    }
    printf(edge_match ? " ✅\n" : " ❌\n");
    
    delete[] I_edge;
    delete[] D_edge;
    
    // Test small k vs large k
    printf("  Testing k=min(10,n_database)...");
    diNoLib::idx_t k_large = std::min((diNoLib::idx_t)10, n_database);
    diNoLib::idx_t *I_large = new diNoLib::idx_t[n_query * k_large];
    float *D_large = new float[n_query * k_large];
    diNoLib::idx_t *I_large2 = new diNoLib::idx_t[n_query * k_large];
    float *D_large2 = new float[n_query * k_large];
    
    bf_search.searchIndex(query, n_query, k_large, I_large, D_large);
    lbbf_search.searchIndex(query, n_query, k_large, I_large2, D_large2);
    
    bool large_match = true;
    for (diNoLib::idx_t qi = 0; qi < n_query * k_large; qi++) {
        if (I_large[qi] != I_large2[qi] || !isClose(D_large[qi], D_large2[qi])) {
            large_match = false;
            break;
        }
    }
    printf(large_match ? " ✅\n" : " ❌\n");
    
    delete[] I_large;
    delete[] D_large;
    delete[] I_large2;
    delete[] D_large2;
    
    // Clean up
    delete[] database;
    delete[] query;
    delete[] I_bf;
    delete[] D_bf;
    delete[] I_lbbf;
    delete[] D_lbbf;
    
    printf("\n=== VERIFICATION COMPLETE ===\n");
    return results_match && edge_match && large_match ? 0 : 1;
}
