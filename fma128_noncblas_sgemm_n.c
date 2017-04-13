#include <string.h>
#include <stdint.h>
#include <immintrin.h>

#define func_name fma128_noncblas_sgemm_n
#define tune_name fma128_noncblas_sgemm_n_tune

typedef float   scalar_t;
typedef __m128  fp_vector_t;
typedef __m128i int_vector_t;
typedef __m128  fp_vector4_t;
typedef __m128i int_vector4_t;

#define MM_FMADD(a, b, c)            _mm_fmadd_ps((a),(b), (c))
#define MM_BROADCAST_Sx(a)           _mm_broadcast_ss((a))
#define MM_ADD_Px(a, b)              _mm_add_ps((a),(b))
#define MM_MUL_Px(a, b)              _mm_mul_ps((a),(b))
#define MM_STOREU_Px(a, b)           _mm_storeu_ps((a),(b))
#define MM_LOADU_Px(a)               _mm_loadu_ps((a))
#define MM_MASKSTOREU_Px(a, mask, b) _mm_maskstore_ps((a),(mask),(b))
#define MM_MASKLOADU_Px(a, mask)     _mm_maskload_ps((a),(mask))
#define MM_SETZERO_Px()              _mm_setzero_ps()

#define MM_LOADU4_Px(a)              _mm_loadu_ps((a))
#define MM_MASKLOADU4_Px(a, mask)    _mm_maskload_ps((a),(mask))

#define USE_CONSTANT_M_STEP
enum {
 N_STEP_MULTIPLIER = 2,
 K_STEP            = 250,
 M_STEP            = 1000,
 SMALL_M_THR       = 56,
 SMALL_M_NxK_STEP  = 24000,
};

#include "avxnnn_noncblas_sgemm_n.c"
