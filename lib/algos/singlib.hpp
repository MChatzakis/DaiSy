#ifndef SINGLIB_HPP
#define SINGLIB_HPP

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned char singlib_sax_t;

void initialdevice(void);
void GPUsyn(void);
void GPUfree(void *devicememorypointer);

float *initialgqts(float *gqts);
singlib_sax_t *initialgsaxarray(singlib_sax_t *gsaxarray, unsigned long datasize);
singlib_sax_t *initialsaxarray(singlib_sax_t *saxarray, unsigned long datasize);
float *initialgposbitmapfloat(float *gposbitmap, unsigned long datasize);
float *initialposbitmapfloat(float *posbitmap, unsigned long datasize);
void gpumemcpy(singlib_sax_t *gsaxarray, const singlib_sax_t *saxarray, unsigned long datasize);

void LBDfloatstreamGPU(singlib_sax_t *saxarray, float *posbitmap, float *qts, float *gqts,
                       float BSF, unsigned long datasize, float *gposbitmap,
                       int segmentnumber, float segmentsize);

#ifdef __cplusplus
}
#endif

#endif 
