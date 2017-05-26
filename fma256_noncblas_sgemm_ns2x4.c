#include <string.h>
#include <stdint.h>
#include <immintrin.h>

#define func_name fma256_noncblas_sgemm_ns2x4
#define tune_name fma256_noncblas_sgemm_ns2x4_tune

typedef float   scalar_t;
typedef __m256  fp_vector_t;
typedef __m256i int_vector_t;
typedef __m128  fp_vector4_t;
typedef __m128i int_vector4_t;

#define MM_FMADD(a, b, c)            _mm256_fmadd_ps((a), (b), (c))
#define MM_BROADCAST_Sx(a)           _mm256_broadcast_ss((a))
#define MM_ADD_Px(a, b)              _mm256_add_ps((a),(b))
#define MM_MUL_Px(a, b)              _mm256_mul_ps((a),(b))
#define MM_STOREU_Px(a, b)           _mm256_storeu_ps((a),(b))
#define MM_LOADU_Px(a)               _mm256_loadu_ps((a))
#define MM_MASKSTOREU_Px(a, mask, b) _mm256_maskstore_ps((a),(mask),(b))
#define MM_MASKLOADU_Px(a, mask)     _mm256_maskload_ps((a),(mask))
#define MM_SETZERO_Px()              _mm256_setzero_ps()

#define MM_LOADU4_Px(a)              _mm_loadu_ps((a))
#define MM_MASKLOADU4_Px(a, mask)    _mm_maskload_ps((a),(mask))

#define USE_CONSTANT_M_STEP
enum {
 N_STEP_MULTIPLIER = 3,
 MxN_BLOCK_SZ      = 1600000,
 K_STEP            = 80,
 M_STEP            = 260,
};

#include "avxnnn_noncblas_sgemm_ns2x4_unroll2.c"
