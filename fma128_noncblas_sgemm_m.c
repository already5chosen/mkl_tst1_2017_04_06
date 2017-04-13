#include <string.h>
#include <stdint.h>
#include <intrin.h>

#define func_name fma128_noncblas_sgemm_m
#define MM_FMADD(a, b, c) _mm_fmadd_ps((a),(b), (c))

typedef float   scalar_t;
typedef __m128  fp_vector_t;
typedef __m128i int_vector_t;

#define MM_BROADCAST_Sx(a)           _mm_broadcast_ss((a))
#define MM_MUL_Px(a, b)              _mm_mul_ps((a),(b))
#define MM_STOREU_Px(a, b)           _mm_storeu_ps((a),(b))
#define MM_LOADU_Px(a)               _mm_loadu_ps((a))
#define MM_MASKSTOREU_Px(a, mask, b) _mm_maskstore_ps((a),(mask),(b))
#define MM_MASKLOADU_Px(a, mask)     _mm_maskload_ps((a),(mask))

enum {
 k_step     = 99,
 m_step_nom = 200,
 m_step_max = 320,
};

#include "avxnnn_noncblas_sgemm_m.c"
