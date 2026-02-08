/*
 * singlib.cu - CUDA kernels and host code for Sing (LBD lower-bound distance).
 * Compilato con nvcc quando SING_CUDA è abilitato.
 */

#include "singlib.hpp"
#include <float.h>
#include <cuda_runtime.h>

/* Per i kernel (singlib_sax_t = unsigned char) */
typedef singlib_sax_t sax_type;

#define streamnumber 20
#define PAA_SEGMENTS_SAX 16  /* usato per size SAX in initialgsaxarray/gpumemcpy; deve coincidere con paa_segments se possibile */

/* Thread per block: multiplo di 32 (warp), adatto a A100 (sm_80) e altre architetture. */
#define LBD_THREADS_PER_BLOCK 256
/* Max blocchi: A100 ha 108 SMs, fino a 32 blocchi/SM; limitiamo a 4096 per non eccedere. */
#define LBD_MAX_BLOCKS 4096

/* --- Init / copy / free (da singlib originale) --- */
extern "C" void initialdevice(void)
{
    cudaSetDevice(0);
}

extern "C" void GPUsyn(void)
{
    cudaDeviceSynchronize();
}

extern "C" void GPUfree(void *devicememorypointer)
{
    cudaFree(devicememorypointer);
}

extern "C" float *initialgqts(float *gqts)
{
    (void)gqts;
    cudaMalloc((void **)&gqts, sizeof(float) * 256);
    return gqts;
}

extern "C" singlib_sax_t *initialgsaxarray(singlib_sax_t *gsaxarray, unsigned long datasize)
{
    (void)gsaxarray;
    cudaMalloc((void **)&gsaxarray, sizeof(singlib_sax_t) * (size_t)datasize * PAA_SEGMENTS_SAX);
    return gsaxarray;
}

extern "C" singlib_sax_t *initialsaxarray(singlib_sax_t *saxarray, unsigned long datasize)
{
    (void)saxarray;
    cudaMallocHost((void **)&saxarray, sizeof(singlib_sax_t) * (size_t)datasize * PAA_SEGMENTS_SAX);
    return saxarray;
}

extern "C" float *initialgposbitmapfloat(float *gposbitmap, unsigned long datasize)
{
    (void)gposbitmap;
    cudaMalloc((void **)&gposbitmap, sizeof(float) * (size_t)datasize);
    return gposbitmap;
}

extern "C" float *initialposbitmapfloat(float *posbitmap, unsigned long datasize)
{
    (void)posbitmap;
    cudaMallocHost((void **)&posbitmap, sizeof(float) * (size_t)datasize);
    return posbitmap;
}

extern "C" void gpumemcpy(singlib_sax_t *gsaxarray, const singlib_sax_t *saxarray, unsigned long datasize)
{
    cudaMemcpy(gsaxarray, saxarray, sizeof(singlib_sax_t) * (size_t)datasize * PAA_SEGMENTS_SAX, cudaMemcpyHostToDevice);
}

/*
 * Kernel calculate_lbdfloat dall'originale: lower-bound distance (LBD) su blocco SAX.
 */
__global__ void calculate_lbdfloat(
    const sax_type * const saxarray,
    const float * const paa,
    const long int M,
    const int N,
    float * positionarray,
    const float BSF,
    float segmentsize)
{
    const int thid = blockDim.x * blockIdx.x + threadIdx.x;
    float distance = 0;

    int i = 0;
    float breakpoint_lower = 0;
    float breakpoint_upper = 0;

    for (int j = thid; j < M; j += gridDim.x * blockDim.x)
    {
        distance = 0;
        for (i = 0; i < N; i++)
        {
            if (segmentsize * distance < BSF)
            {
                sax_type v = saxarray[j * N + i];
                sax_type region_lower = v;
                sax_type region_upper = ((sax_type)(~0) | region_lower);

                if (region_lower == 0)
                {
                    breakpoint_lower = -2000000;
                    float breaku = ((float)region_lower - 127.0f) / 128.0f;
                    breakpoint_upper = breaku * (1.1362582192f * breaku * breaku + 0.99800f);
                    if (breakpoint_upper < paa[i])
                    {
                        distance += (breakpoint_upper - paa[i]) * (breakpoint_upper - paa[i]);
                    }
                }
                else if (region_upper == 256 - 1)
                {
                    breakpoint_upper = +2000000;
                    float breakx = ((float)region_lower - 128.0f) / 128.0f;
                    breakpoint_lower = breakx * (breakx * breakx * 1.1362582192f + 0.99800f);
                    if (breakpoint_lower > paa[i])
                    {
                        distance += (breakpoint_lower - paa[i]) * (breakpoint_lower - paa[i]);
                    }
                }
                else
                {
                    float breakx = ((float)region_lower - 128.0f) / 128.0f;
                    breakpoint_lower = breakx * (breakx * breakx * 1.1362582192f + 0.99800f);
                    if (breakpoint_lower > paa[i])
                    {
                        distance += (breakpoint_lower - paa[i]) * (breakpoint_lower - paa[i]);
                    }
                    else
                    {
                        float breaku = ((float)region_lower - 127.0f) / 128.0f;
                        breakpoint_upper = breaku * (1.1362582192f * breaku * breaku + 0.99800f);
                        if (breakpoint_upper < paa[i])
                        {
                            distance += (breakpoint_upper - paa[i]) * (breakpoint_upper - paa[i]);
                        }
                    }
                }
            }
        }
        positionarray[j] = segmentsize * distance;
    }
}

extern "C" void LBDfloatstreamGPU(
    singlib_sax_t *saxarray,
    float *posbitmap,
    float *qts,
    float *gqts,
    float BSF,
    unsigned long datasize,
    float *gposbitmap,
    int segmentnumber,
    float segmentsize)
{
    cudaMemcpy(gqts, qts, sizeof(float) * (size_t)segmentnumber, cudaMemcpyHostToDevice);

    cudaStream_t streams[streamnumber];
    for (int i = 0; i < streamnumber; i++)
    {
        cudaStreamCreate(&streams[i]);
    }

    for (int i = 0; i < streamnumber; i++)
    {
        long int M = (long int)(datasize / streamnumber);
        /* Grid size: coprire M elementi; almeno 1 blocco, al massimo LBD_MAX_BLOCKS (adatto a A100 e simili). */
        int num_blocks = (M <= 0) ? 1 : (int)((M + (long int)LBD_THREADS_PER_BLOCK - 1) / (long int)LBD_THREADS_PER_BLOCK);
        if (num_blocks > LBD_MAX_BLOCKS)
            num_blocks = LBD_MAX_BLOCKS;
        if (num_blocks < 1)
            num_blocks = 1;

        calculate_lbdfloat<<<num_blocks, LBD_THREADS_PER_BLOCK, 20, streams[i]>>>(
            saxarray + i * datasize * segmentnumber / streamnumber,
            gqts,
            M,
            segmentnumber,
            gposbitmap + i * datasize / streamnumber,
            BSF,
            segmentsize);
        cudaMemcpyAsync(
            posbitmap + i * datasize / streamnumber,
            gposbitmap + i * datasize / streamnumber,
            sizeof(float) * (size_t)(datasize / streamnumber),
            cudaMemcpyDeviceToHost,
            streams[i]);
    }

    cudaDeviceSynchronize();
}
