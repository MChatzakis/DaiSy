#include "test_utils.hpp" 

diNoLib::DistanceType dist_L2Squared = diNoLib::DistanceType::L2_SQUARED;
diNoLib::BruteForceSearch* search = new diNoLib::BruteForceSearch(dist_L2Squared);

/*******************************************************/
/************************ ASTRO ************************/
/*******************************************************/
const char *astro_data = "../data/astronomy.data.len256.size50000.znorm.bin";
const char *astro_query = "../data/astronomy.query.len256.size100.znorm.bin";

/************************ THREAD 1 ************************/
TEST_F(SimilaritySearchTest, BruteForceSSTest_AstronomyData_q100_k1_thread1) 
{
    runSST(
        search,
        "../tests/gt/Indices/bruteFSS_gt_I_astronomy_len256_size50000_q100_k1.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_astronomy_len256_size50000_q100_k1.txt",
        astro_data,
        astro_query
    );
}

TEST_F(SimilaritySearchTest, BruteForceSSTest_AstronomyData_q100_k10_thread1) 
{
    runSST(
        search,
        "../tests/gt/Indices/bruteFSS_gt_I_astronomy_len256_size50000_q100_k10.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_astronomy_len256_size50000_q100_k10.txt",
        astro_data,
        astro_query
    );
}

TEST_F(SimilaritySearchTest, BruteForceSSTest_AstronomyData_q100_k100_thread1) 
{
    runSST(
        search,
        "../tests/gt/Indices/bruteFSS_gt_I_astronomy_len256_size50000_q100_k100.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_astronomy_len256_size50000_q100_k100.txt",
        astro_data,
        astro_query
    );
}

/************************ THREAD 4 ************************/
TEST_F(SimilaritySearchTest, BruteForceSSTest_AstronomyData_q100_k1_thread4) 
{
    runSST(
        search,
        "../tests/gt/Indices/bruteFSS_gt_I_astronomy_len256_size50000_q100_k1.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_astronomy_len256_size50000_q100_k1.txt",
        astro_data,
        astro_query,
        4
    );
}

TEST_F(SimilaritySearchTest, BruteForceSSTest_AstronomyData_q100_k10_thread4) 
{
    runSST(
        search,
        "../tests/gt/Indices/bruteFSS_gt_I_astronomy_len256_size50000_q100_k10.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_astronomy_len256_size50000_q100_k10.txt",
        astro_data,
        astro_query,
        4
    );
}

TEST_F(SimilaritySearchTest, BruteForceSSTest_AstronomyData_q100_k100_thread4) 
{
    runSST(
        search,
        "../tests/gt/Indices/bruteFSS_gt_I_astronomy_len256_size50000_q100_k100.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_astronomy_len256_size50000_q100_k100.txt",
        astro_data,
        astro_query,
        4
    );
}

/************************ THREAD 8  ************************/
TEST_F(SimilaritySearchTest, BruteForceSSTest_AstronomyData_q100_k1_thread8) 
{
    runSST(
        search,
        "../tests/gt/Indices/bruteFSS_gt_I_astronomy_len256_size50000_q100_k1.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_astronomy_len256_size50000_q100_k1.txt",
        astro_data,
        astro_query,
        8
    );
}

TEST_F(SimilaritySearchTest, BruteForceSSTest_AstronomyData_q100_k10_thread8) 
{
    runSST(
        search,
        "../tests/gt/Indices/bruteFSS_gt_I_astronomy_len256_size50000_q100_k10.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_astronomy_len256_size50000_q100_k10.txt",
        astro_data,
        astro_query,
        8
    );
}

TEST_F(SimilaritySearchTest, BruteForceSSTest_AstronomyData_q100_k100_thread8) 
{
    runSST(
        search,
        "../tests/gt/Indices/bruteFSS_gt_I_astronomy_len256_size50000_q100_k100.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_astronomy_len256_size50000_q100_k100.txt",
        astro_data,
        astro_query,
        8
    );
}

/********************************************************/
/************************ RANDOM ************************/
/********************************************************/
const char *random_data = "../data/random.data.randwalk.len96.size200000.znorm.bin";
const char *random_query = "../data/random.query.randwalk.len96.size1000.bin";

/************************ THREAD 1 ************************/
TEST_F(SimilaritySearchTest, BruteForceSSTest_RandomWalkData_q1000_k1_thread1) 
{
    runSST(
        search,
        "../tests/gt/Indices/bruteFSS_gt_I_random_len96_size200000_q1000_k1.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_random_len96_size200000_q1000_k1.txt",
        random_data,
        random_query
    );
}

TEST_F(SimilaritySearchTest, BruteForceSSTest_RandomWalkData_q1000_k10_thread1) 
{
    runSST(
        search,
        "../tests/gt/Indices/bruteFSS_gt_I_random_len96_size200000_q1000_k10.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_random_len96_size200000_q1000_k10.txt",
        random_data,
        random_query
    );
}

TEST_F(SimilaritySearchTest, BruteForceSSTest_RandomWalkData_q1000_k100_thread1) 
{
    runSST(
        search,
        "../tests/gt/Indices/bruteFSS_gt_I_random_len96_size200000_q1000_k100.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_random_len96_size200000_q1000_k100.txt",
        random_data,
        random_query
    );
}

/************************ THREAD 4 ************************/
TEST_F(SimilaritySearchTest, BruteForceSSTest_RandomWalkData_q1000_k1_thread4) 
{
    runSST(
        search,
        "../tests/gt/Indices/bruteFSS_gt_I_random_len96_size200000_q1000_k1.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_random_len96_size200000_q1000_k1.txt",
        random_data,
        random_query,
        4
    );
}

TEST_F(SimilaritySearchTest, BruteForceSSTest_RandomWalkData_q1000_k10_thread4) 
{
    runSST(
        search,
        "../tests/gt/Indices/bruteFSS_gt_I_random_len96_size200000_q1000_k10.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_random_len96_size200000_q1000_k10.txt",
        random_data,
        random_query,
        4
    );
}

TEST_F(SimilaritySearchTest, BruteForceSSTest_RandomWalkData_q1000_k100_thread4) 
{
    runSST(
        search,
        "../tests/gt/Indices/bruteFSS_gt_I_random_len96_size200000_q1000_k100.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_random_len96_size200000_q1000_k100.txt",
        random_data,
        random_query,
        4
    );
}

/************************ THREAD 8 ************************/
TEST_F(SimilaritySearchTest, BruteForceSSTest_RandomWalkData_q1000_k1_thread8) 
{
    runSST(
        search,
        "../tests/gt/Indices/bruteFSS_gt_I_random_len96_size200000_q1000_k1.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_random_len96_size200000_q1000_k1.txt",
        random_data,
        random_query,
        8
    );
}

TEST_F(SimilaritySearchTest, BruteForceSSTest_RandomWalkData_q1000_k10_thread8) 
{
    runSST(
        search,
        "../tests/gt/Indices/bruteFSS_gt_I_random_len96_size200000_q1000_k10.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_random_len96_size200000_q1000_k10.txt",
        random_data,
        random_query,
        8
    );
}

TEST_F(SimilaritySearchTest, BruteForceSSTest_RandomWalkData_q1000_k100_thread8) 
{
    runSST(
        search,
        "../tests/gt/Indices/bruteFSS_gt_I_random_len96_size200000_q1000_k100.txt",
        "../tests/gt/Distances/bruteFSS_gt_D_random_len96_size200000_q1000_k100.txt",
        random_data,
        random_query,
        8
    );
}

int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}