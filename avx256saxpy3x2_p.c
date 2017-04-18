#include <string.h>
#include <stdint.h>
#include <immintrin.h>

#define func_name avx256_noncblas_sgemm_p

typedef float   scalar_t;
typedef __m256  fp_vector_t;
typedef __m256i int_vector_t;
typedef __m128  fp_vector4_t;
typedef __m128i int_vector4_t;

#define MM_FMADD(a, b, c)            _mm256_add_ps(_mm256_mul_ps((a),(b)), (c))
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


enum {
 SIMD_FACTOR          = sizeof(fp_vector_t)/sizeof(scalar_t),
};


void saxpy_3x2(
  const scalar_t* A, unsigned lda,
  const scalar_t* B, unsigned ldb,
  scalar_t*       C, unsigned ldc,
  unsigned        loopCnt,
  unsigned cnt,  unsigned cntMsk)
{
  const unsigned ldb1 = ldb;
  const unsigned ldb2 = ldb1+ldb1;
  //const unsigned ldb3 = ldb2+ldb1;
  //const unsigned ldb4 = ldb3+ldb1;
  // const int ldb5 = ldb4+ldb1;
  for (unsigned i = 0; i < cnt; ++i) {
    unsigned ix      = i & cntMsk;

    const scalar_t* Arow = &A[lda*2*ix];
    scalar_t*       Crow = &C[ldc*2*ix];

    fp_vector_t a00 = MM_BROADCAST_Sx(&Arow[0]);
    fp_vector_t a01 = MM_BROADCAST_Sx(&Arow[1]);
    fp_vector_t a02 = MM_BROADCAST_Sx(&Arow[2]);
    Arow += lda;
    fp_vector_t a10 = MM_BROADCAST_Sx(&Arow[0]);
    fp_vector_t a11 = MM_BROADCAST_Sx(&Arow[1]);
    fp_vector_t a12 = MM_BROADCAST_Sx(&Arow[2]);
    Arow += lda;

    const scalar_t *Brow = B;
    unsigned n = loopCnt;
    do {
      fp_vector_t c0 = MM_LOADU_Px(&Crow[0]);
      fp_vector_t c1 = MM_LOADU_Px(&Crow[ldc]);
      fp_vector_t b;

      b = MM_LOADU_Px(&Brow[0]);
      c0 = MM_FMADD(b, a00, c0);
      c1 = MM_FMADD(b, a10, c1);

      b = MM_LOADU_Px(&Brow[ldb1]);
      c0 = MM_FMADD(b, a01, c0);
      c1 = MM_FMADD(b, a11, c1);

      b = MM_LOADU_Px(&Brow[ldb2]);
      c0 = MM_FMADD(b, a02, c0);
      c1 = MM_FMADD(b, a12, c1);

      MM_STOREU_Px(&Crow[0],   c0);
      MM_STOREU_Px(&Crow[ldc], c1);
      Brow += SIMD_FACTOR;
      Crow += SIMD_FACTOR;
    } while (--n);
  }
}
