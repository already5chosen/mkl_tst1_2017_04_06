#include <string.h>
#include <stdint.h>
#include <immintrin.h>

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
 SIMD_FACTOR  = sizeof(fp_vector_t)/sizeof(scalar_t),
};

typedef struct {
  int_vector_t mask_n;
  const scalar_t *A;
  const scalar_t *B;
  scalar_t *C;
  scalar_t alpha;
  scalar_t beta;
  int M, N, K, lda, ldb, ldc;
} gemm_prm_t;

// K < 8
static void avx256_noncblas_sgemm_smallK(gemm_prm_t* pPrm)
{
  // cases of very small K
  fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
  fp_vector_t beta_ps  = MM_BROADCAST_Sx(&pPrm->beta);
  int N = pPrm->N;
  int nw   = (unsigned)N / SIMD_FACTOR;
  int nRem = (unsigned)N % SIMD_FACTOR;
  const scalar_t *A = pPrm->A;
  const int lda     = pPrm->lda;
  scalar_t *C       = pPrm->C;
  const int ldc     = pPrm->ldc;
  int K = pPrm->K;
  if (K == 7) {
    const int ldb1 = pPrm->ldb;
    const int ldb2 = ldb1+ldb1;
    const int ldb3 = ldb2+ldb1;
    const int ldb4 = ldb3+ldb1;
    const int ldb5 = ldb4+ldb1;
    const int ldb6 = ldb5+ldb1;
    for (unsigned mh = (unsigned)pPrm->M / 2; mh != 0; --mh) {
      const scalar_t *bb = pPrm->B;
      scalar_t *cc = C;
      for (unsigned ni = nw; ni != 0; bb += SIMD_FACTOR, cc += SIMD_FACTOR, --ni) {
        fp_vector_t a0, a1, b;

        a0 = MM_BROADCAST_Sx(&A[    0]);
        a1 = MM_BROADCAST_Sx(&A[lda+0]);
        b = MM_LOADU_Px(&bb[0]);
        fp_vector_t acc0 = MM_MUL_Px(b, a0);
        fp_vector_t acc1 = MM_MUL_Px(b, a1);

        a0 = MM_BROADCAST_Sx(&A[    1]);
        a1 = MM_BROADCAST_Sx(&A[lda+1]);
        b = MM_LOADU_Px(&bb[ldb1]);
        acc0 = MM_FMADD(b, a0, acc0);
        acc1 = MM_FMADD(b, a1, acc1);

        a0 = MM_BROADCAST_Sx(&A[    2]);
        a1 = MM_BROADCAST_Sx(&A[lda+2]);
        b = MM_LOADU_Px(&bb[ldb2]);
        acc0 = MM_FMADD(b, a0, acc0);
        acc1 = MM_FMADD(b, a1, acc1);

        a0 = MM_BROADCAST_Sx(&A[    3]);
        a1 = MM_BROADCAST_Sx(&A[lda+3]);
        b = MM_LOADU_Px(&bb[ldb3]);
        acc0 = MM_FMADD(b, a0, acc0);
        acc1 = MM_FMADD(b, a1, acc1);

        a0 = MM_BROADCAST_Sx(&A[    4]);
        a1 = MM_BROADCAST_Sx(&A[lda+4]);
        b = MM_LOADU_Px(&bb[ldb4]);
        acc0 = MM_FMADD(b, a0, acc0);
        acc1 = MM_FMADD(b, a1, acc1);

        a0 = MM_BROADCAST_Sx(&A[    5]);
        a1 = MM_BROADCAST_Sx(&A[lda+5]);
        b = MM_LOADU_Px(&bb[ldb5]);
        acc0 = MM_FMADD(b, a0, acc0);
        acc1 = MM_FMADD(b, a1, acc1);

        a0 = MM_BROADCAST_Sx(&A[    6]);
        a1 = MM_BROADCAST_Sx(&A[lda+6]);
        b = MM_LOADU_Px(&bb[ldb6]);
        acc0 = MM_FMADD(b, a0, acc0);
        acc1 = MM_FMADD(b, a1, acc1);

        fp_vector_t c0 = MM_MUL_Px(MM_LOADU_Px(&cc[0])  , beta_ps);
        fp_vector_t c1 = MM_MUL_Px(MM_LOADU_Px(&cc[ldc]), beta_ps);
        MM_STOREU_Px(&cc[0]  , MM_FMADD(acc0, alpha_ps, c0));
        MM_STOREU_Px(&cc[ldc], MM_FMADD(acc1, alpha_ps, c1));
      }
      if (nRem) {
        fp_vector_t a0, a1, b;
        int_vector_t mask_n = pPrm->mask_n;

        a0 = MM_BROADCAST_Sx(&A[    0]);
        a1 = MM_BROADCAST_Sx(&A[lda+0]);
        b = MM_MASKLOADU_Px(&bb[0], mask_n);
        fp_vector_t acc0 = MM_MUL_Px(b, a0);
        fp_vector_t acc1 = MM_MUL_Px(b, a1);

        a0 = MM_BROADCAST_Sx(&A[    1]);
        a1 = MM_BROADCAST_Sx(&A[lda+1]);
        b = MM_MASKLOADU_Px(&bb[ldb1], mask_n);
        acc0 = MM_FMADD(b, a0, acc0);
        acc1 = MM_FMADD(b, a1, acc1);

        a0 = MM_BROADCAST_Sx(&A[    2]);
        a1 = MM_BROADCAST_Sx(&A[lda+2]);
        b = MM_MASKLOADU_Px(&bb[ldb2], mask_n);
        acc0 = MM_FMADD(b, a0, acc0);
        acc1 = MM_FMADD(b, a1, acc1);

        a0 = MM_BROADCAST_Sx(&A[    3]);
        a1 = MM_BROADCAST_Sx(&A[lda+3]);
        b = MM_MASKLOADU_Px(&bb[ldb3], mask_n);
        acc0 = MM_FMADD(b, a0, acc0);
        acc1 = MM_FMADD(b, a1, acc1);

        a0 = MM_BROADCAST_Sx(&A[    4]);
        a1 = MM_BROADCAST_Sx(&A[lda+4]);
        b = MM_MASKLOADU_Px(&bb[ldb4], mask_n);
        acc0 = MM_FMADD(b, a0, acc0);
        acc1 = MM_FMADD(b, a1, acc1);

        a0 = MM_BROADCAST_Sx(&A[    5]);
        a1 = MM_BROADCAST_Sx(&A[lda+5]);
        b = MM_MASKLOADU_Px(&bb[ldb5], mask_n);
        acc0 = MM_FMADD(b, a0, acc0);
        acc1 = MM_FMADD(b, a1, acc1);

        a0 = MM_BROADCAST_Sx(&A[    6]);
        a1 = MM_BROADCAST_Sx(&A[lda+6]);
        b = MM_MASKLOADU_Px(&bb[ldb6], mask_n);
        acc0 = MM_FMADD(b, a0, acc0);
        acc1 = MM_FMADD(b, a1, acc1);

        fp_vector_t c0 = MM_MUL_Px(MM_MASKLOADU_Px(&cc[0],   mask_n), beta_ps);
        fp_vector_t c1 = MM_MUL_Px(MM_MASKLOADU_Px(&cc[ldc], mask_n), beta_ps);
        MM_MASKSTOREU_Px(&cc[0]  , mask_n, MM_FMADD(acc0, alpha_ps, c0));
        MM_MASKSTOREU_Px(&cc[ldc], mask_n, MM_FMADD(acc1, alpha_ps, c1));
      }
      A += lda*2;
      C += ldc*2;
    }
    if ((unsigned)pPrm->M % 2) {
      const scalar_t *bb = pPrm->B;
      scalar_t *cc = C;
      for (unsigned ni = nw; ni != 0; bb += SIMD_FACTOR, cc += SIMD_FACTOR, --ni) {
        fp_vector_t a0, b;

        a0 = MM_BROADCAST_Sx(&A[    0]);
        b = MM_LOADU_Px(&bb[0]);
        fp_vector_t acc0 = MM_MUL_Px(b, a0);

        a0 = MM_BROADCAST_Sx(&A[    1]);
        b = MM_LOADU_Px(&bb[ldb1]);
        acc0 = MM_FMADD(b, a0, acc0);

        a0 = MM_BROADCAST_Sx(&A[    2]);
        b = MM_LOADU_Px(&bb[ldb2]);
        acc0 = MM_FMADD(b, a0, acc0);

        a0 = MM_BROADCAST_Sx(&A[    3]);
        b = MM_LOADU_Px(&bb[ldb3]);
        acc0 = MM_FMADD(b, a0, acc0);

        a0 = MM_BROADCAST_Sx(&A[    4]);
        b = MM_LOADU_Px(&bb[ldb4]);
        acc0 = MM_FMADD(b, a0, acc0);

        a0 = MM_BROADCAST_Sx(&A[    5]);
        b = MM_LOADU_Px(&bb[ldb5]);
        acc0 = MM_FMADD(b, a0, acc0);

        a0 = MM_BROADCAST_Sx(&A[    6]);
        b = MM_LOADU_Px(&bb[ldb6]);
        acc0 = MM_FMADD(b, a0, acc0);

        fp_vector_t c0 = MM_MUL_Px(MM_LOADU_Px(&cc[0])  , beta_ps);
        MM_STOREU_Px(&cc[0]  , MM_FMADD(acc0, alpha_ps, c0));
      }
      if (nRem) {
        fp_vector_t a0, b;
        int_vector_t mask_n = pPrm->mask_n;

        a0 = MM_BROADCAST_Sx(&A[    0]);
        b = MM_MASKLOADU_Px(&bb[0], mask_n);
        fp_vector_t acc0 = MM_MUL_Px(b, a0);

        a0 = MM_BROADCAST_Sx(&A[    1]);
        b = MM_MASKLOADU_Px(&bb[ldb1], mask_n);
        acc0 = MM_FMADD(b, a0, acc0);

        a0 = MM_BROADCAST_Sx(&A[    2]);
        b = MM_MASKLOADU_Px(&bb[ldb2], mask_n);
        acc0 = MM_FMADD(b, a0, acc0);

        a0 = MM_BROADCAST_Sx(&A[    3]);
        b = MM_MASKLOADU_Px(&bb[ldb3], mask_n);
        acc0 = MM_FMADD(b, a0, acc0);

        a0 = MM_BROADCAST_Sx(&A[    4]);
        b = MM_MASKLOADU_Px(&bb[ldb4], mask_n);
        acc0 = MM_FMADD(b, a0, acc0);

        a0 = MM_BROADCAST_Sx(&A[    5]);
        b = MM_MASKLOADU_Px(&bb[ldb5], mask_n);
        acc0 = MM_FMADD(b, a0, acc0);

        a0 = MM_BROADCAST_Sx(&A[    6]);
        b = MM_MASKLOADU_Px(&bb[ldb6], mask_n);
        acc0 = MM_FMADD(b, a0, acc0);

        fp_vector_t c0 = MM_MUL_Px(MM_MASKLOADU_Px(&cc[0],   mask_n), beta_ps);
        MM_MASKSTOREU_Px(&cc[0]  , mask_n, MM_FMADD(acc0, alpha_ps, c0));
      }
    }
  } else if (K == 6) {
    const int ldb1 = pPrm->ldb;
    const int ldb2 = ldb1+ldb1;
    const int ldb3 = ldb2+ldb1;
    const int ldb4 = ldb3+ldb1;
    const int ldb5 = ldb4+ldb1;
    for (unsigned mh = (unsigned)pPrm->M / 2; mh != 0; --mh) {
      const scalar_t *bb = pPrm->B;
      scalar_t *cc = C;
      for (unsigned ni = nw; ni != 0; bb += SIMD_FACTOR, cc += SIMD_FACTOR, --ni) {
        fp_vector_t a0, a1, b;

        a0 = MM_BROADCAST_Sx(&A[    0]);
        a1 = MM_BROADCAST_Sx(&A[lda+0]);
        b = MM_LOADU_Px(&bb[0]);
        fp_vector_t acc0 = MM_MUL_Px(b, a0);
        fp_vector_t acc1 = MM_MUL_Px(b, a1);

        a0 = MM_BROADCAST_Sx(&A[    1]);
        a1 = MM_BROADCAST_Sx(&A[lda+1]);
        b = MM_LOADU_Px(&bb[ldb1]);
        acc0 = MM_FMADD(b, a0, acc0);
        acc1 = MM_FMADD(b, a1, acc1);

        a0 = MM_BROADCAST_Sx(&A[    2]);
        a1 = MM_BROADCAST_Sx(&A[lda+2]);
        b = MM_LOADU_Px(&bb[ldb2]);
        acc0 = MM_FMADD(b, a0, acc0);
        acc1 = MM_FMADD(b, a1, acc1);

        a0 = MM_BROADCAST_Sx(&A[    3]);
        a1 = MM_BROADCAST_Sx(&A[lda+3]);
        b = MM_LOADU_Px(&bb[ldb3]);
        acc0 = MM_FMADD(b, a0, acc0);
        acc1 = MM_FMADD(b, a1, acc1);

        a0 = MM_BROADCAST_Sx(&A[    4]);
        a1 = MM_BROADCAST_Sx(&A[lda+4]);
        b = MM_LOADU_Px(&bb[ldb4]);
        acc0 = MM_FMADD(b, a0, acc0);
        acc1 = MM_FMADD(b, a1, acc1);

        a0 = MM_BROADCAST_Sx(&A[    5]);
        a1 = MM_BROADCAST_Sx(&A[lda+5]);
        b = MM_LOADU_Px(&bb[ldb5]);
        acc0 = MM_FMADD(b, a0, acc0);
        acc1 = MM_FMADD(b, a1, acc1);

        fp_vector_t c0 = MM_MUL_Px(MM_LOADU_Px(&cc[0])  , beta_ps);
        fp_vector_t c1 = MM_MUL_Px(MM_LOADU_Px(&cc[ldc]), beta_ps);
        MM_STOREU_Px(&cc[0]  , MM_FMADD(acc0, alpha_ps, c0));
        MM_STOREU_Px(&cc[ldc], MM_FMADD(acc1, alpha_ps, c1));
      }
      if (nRem) {
        fp_vector_t a0, a1, b;
        int_vector_t mask_n = pPrm->mask_n;

        a0 = MM_BROADCAST_Sx(&A[    0]);
        a1 = MM_BROADCAST_Sx(&A[lda+0]);
        b = MM_MASKLOADU_Px(&bb[0], mask_n);
        fp_vector_t acc0 = MM_MUL_Px(b, a0);
        fp_vector_t acc1 = MM_MUL_Px(b, a1);

        a0 = MM_BROADCAST_Sx(&A[    1]);
        a1 = MM_BROADCAST_Sx(&A[lda+1]);
        b = MM_MASKLOADU_Px(&bb[ldb1], mask_n);
        acc0 = MM_FMADD(b, a0, acc0);
        acc1 = MM_FMADD(b, a1, acc1);

        a0 = MM_BROADCAST_Sx(&A[    2]);
        a1 = MM_BROADCAST_Sx(&A[lda+2]);
        b = MM_MASKLOADU_Px(&bb[ldb2], mask_n);
        acc0 = MM_FMADD(b, a0, acc0);
        acc1 = MM_FMADD(b, a1, acc1);

        a0 = MM_BROADCAST_Sx(&A[    3]);
        a1 = MM_BROADCAST_Sx(&A[lda+3]);
        b = MM_MASKLOADU_Px(&bb[ldb3], mask_n);
        acc0 = MM_FMADD(b, a0, acc0);
        acc1 = MM_FMADD(b, a1, acc1);

        a0 = MM_BROADCAST_Sx(&A[    4]);
        a1 = MM_BROADCAST_Sx(&A[lda+4]);
        b = MM_MASKLOADU_Px(&bb[ldb4], mask_n);
        acc0 = MM_FMADD(b, a0, acc0);
        acc1 = MM_FMADD(b, a1, acc1);

        a0 = MM_BROADCAST_Sx(&A[    5]);
        a1 = MM_BROADCAST_Sx(&A[lda+5]);
        b = MM_MASKLOADU_Px(&bb[ldb5], mask_n);
        acc0 = MM_FMADD(b, a0, acc0);
        acc1 = MM_FMADD(b, a1, acc1);

        fp_vector_t c0 = MM_MUL_Px(MM_MASKLOADU_Px(&cc[0],   mask_n), beta_ps);
        fp_vector_t c1 = MM_MUL_Px(MM_MASKLOADU_Px(&cc[ldc], mask_n), beta_ps);
        MM_MASKSTOREU_Px(&cc[0]  , mask_n, MM_FMADD(acc0, alpha_ps, c0));
        MM_MASKSTOREU_Px(&cc[ldc], mask_n, MM_FMADD(acc1, alpha_ps, c1));
      }
      A += lda*2;
      C += ldc*2;
    }
    if ((unsigned)pPrm->M % 2) {
      const scalar_t *bb = pPrm->B;
      scalar_t *cc = C;
      for (unsigned ni = nw; ni != 0; bb += SIMD_FACTOR, cc += SIMD_FACTOR, --ni) {
        fp_vector_t a0, b;

        a0 = MM_BROADCAST_Sx(&A[    0]);
        b = MM_LOADU_Px(&bb[0]);
        fp_vector_t acc0 = MM_MUL_Px(b, a0);

        a0 = MM_BROADCAST_Sx(&A[    1]);
        b = MM_LOADU_Px(&bb[ldb1]);
        acc0 = MM_FMADD(b, a0, acc0);

        a0 = MM_BROADCAST_Sx(&A[    2]);
        b = MM_LOADU_Px(&bb[ldb2]);
        acc0 = MM_FMADD(b, a0, acc0);

        a0 = MM_BROADCAST_Sx(&A[    3]);
        b = MM_LOADU_Px(&bb[ldb3]);
        acc0 = MM_FMADD(b, a0, acc0);

        a0 = MM_BROADCAST_Sx(&A[    4]);
        b = MM_LOADU_Px(&bb[ldb4]);
        acc0 = MM_FMADD(b, a0, acc0);

        a0 = MM_BROADCAST_Sx(&A[    5]);
        b = MM_LOADU_Px(&bb[ldb5]);
        acc0 = MM_FMADD(b, a0, acc0);

        fp_vector_t c0 = MM_MUL_Px(MM_LOADU_Px(&cc[0])  , beta_ps);
        MM_STOREU_Px(&cc[0]  , MM_FMADD(acc0, alpha_ps, c0));
      }
      if (nRem) {
        fp_vector_t a0, b;
        int_vector_t mask_n = pPrm->mask_n;

        a0 = MM_BROADCAST_Sx(&A[    0]);
        b = MM_MASKLOADU_Px(&bb[0], mask_n);
        fp_vector_t acc0 = MM_MUL_Px(b, a0);

        a0 = MM_BROADCAST_Sx(&A[    1]);
        b = MM_MASKLOADU_Px(&bb[ldb1], mask_n);
        acc0 = MM_FMADD(b, a0, acc0);

        a0 = MM_BROADCAST_Sx(&A[    2]);
        b = MM_MASKLOADU_Px(&bb[ldb2], mask_n);
        acc0 = MM_FMADD(b, a0, acc0);

        a0 = MM_BROADCAST_Sx(&A[    3]);
        b = MM_MASKLOADU_Px(&bb[ldb3], mask_n);
        acc0 = MM_FMADD(b, a0, acc0);

        a0 = MM_BROADCAST_Sx(&A[    4]);
        b = MM_MASKLOADU_Px(&bb[ldb4], mask_n);
        acc0 = MM_FMADD(b, a0, acc0);

        a0 = MM_BROADCAST_Sx(&A[    5]);
        b = MM_MASKLOADU_Px(&bb[ldb5], mask_n);
        acc0 = MM_FMADD(b, a0, acc0);

        fp_vector_t c0 = MM_MUL_Px(MM_MASKLOADU_Px(&cc[0],   mask_n), beta_ps);
        MM_MASKSTOREU_Px(&cc[0]  , mask_n, MM_FMADD(acc0, alpha_ps, c0));
      }
    }
  } else if (K == 5) {
    const int ldb1 = pPrm->ldb;
    const int ldb2 = ldb1+ldb1;
    const int ldb3 = ldb2+ldb1;
    const int ldb4 = ldb3+ldb1;
    for (unsigned mh = (unsigned)pPrm->M / 2; mh != 0; --mh) {
      const scalar_t *bb = pPrm->B;
      scalar_t *cc = C;
      for (unsigned ni = nw; ni != 0; bb += SIMD_FACTOR, cc += SIMD_FACTOR, --ni) {
        fp_vector_t a0, a1, b;

        a0 = MM_BROADCAST_Sx(&A[    0]);
        a1 = MM_BROADCAST_Sx(&A[lda+0]);
        b = MM_LOADU_Px(&bb[0]);
        fp_vector_t acc0 = MM_MUL_Px(b, a0);
        fp_vector_t acc1 = MM_MUL_Px(b, a1);

        a0 = MM_BROADCAST_Sx(&A[    1]);
        a1 = MM_BROADCAST_Sx(&A[lda+1]);
        b = MM_LOADU_Px(&bb[ldb1]);
        acc0 = MM_FMADD(b, a0, acc0);
        acc1 = MM_FMADD(b, a1, acc1);

        a0 = MM_BROADCAST_Sx(&A[    2]);
        a1 = MM_BROADCAST_Sx(&A[lda+2]);
        b = MM_LOADU_Px(&bb[ldb2]);
        acc0 = MM_FMADD(b, a0, acc0);
        acc1 = MM_FMADD(b, a1, acc1);

        a0 = MM_BROADCAST_Sx(&A[    3]);
        a1 = MM_BROADCAST_Sx(&A[lda+3]);
        b = MM_LOADU_Px(&bb[ldb3]);
        acc0 = MM_FMADD(b, a0, acc0);
        acc1 = MM_FMADD(b, a1, acc1);

        a0 = MM_BROADCAST_Sx(&A[    4]);
        a1 = MM_BROADCAST_Sx(&A[lda+4]);
        b = MM_LOADU_Px(&bb[ldb4]);
        acc0 = MM_FMADD(b, a0, acc0);
        acc1 = MM_FMADD(b, a1, acc1);

        fp_vector_t c0 = MM_MUL_Px(MM_LOADU_Px(&cc[0])  , beta_ps);
        fp_vector_t c1 = MM_MUL_Px(MM_LOADU_Px(&cc[ldc]), beta_ps);
        MM_STOREU_Px(&cc[0]  , MM_FMADD(acc0, alpha_ps, c0));
        MM_STOREU_Px(&cc[ldc], MM_FMADD(acc1, alpha_ps, c1));
      }
      if (nRem) {
        fp_vector_t a0, a1, b;
        int_vector_t mask_n = pPrm->mask_n;

        a0 = MM_BROADCAST_Sx(&A[    0]);
        a1 = MM_BROADCAST_Sx(&A[lda+0]);
        b = MM_MASKLOADU_Px(&bb[0], mask_n);
        fp_vector_t acc0 = MM_MUL_Px(b, a0);
        fp_vector_t acc1 = MM_MUL_Px(b, a1);

        a0 = MM_BROADCAST_Sx(&A[    1]);
        a1 = MM_BROADCAST_Sx(&A[lda+1]);
        b = MM_MASKLOADU_Px(&bb[ldb1], mask_n);
        acc0 = MM_FMADD(b, a0, acc0);
        acc1 = MM_FMADD(b, a1, acc1);

        a0 = MM_BROADCAST_Sx(&A[    2]);
        a1 = MM_BROADCAST_Sx(&A[lda+2]);
        b = MM_MASKLOADU_Px(&bb[ldb2], mask_n);
        acc0 = MM_FMADD(b, a0, acc0);
        acc1 = MM_FMADD(b, a1, acc1);

        a0 = MM_BROADCAST_Sx(&A[    3]);
        a1 = MM_BROADCAST_Sx(&A[lda+3]);
        b = MM_MASKLOADU_Px(&bb[ldb3], mask_n);
        acc0 = MM_FMADD(b, a0, acc0);
        acc1 = MM_FMADD(b, a1, acc1);

        a0 = MM_BROADCAST_Sx(&A[    4]);
        a1 = MM_BROADCAST_Sx(&A[lda+4]);
        b = MM_MASKLOADU_Px(&bb[ldb4], mask_n);
        acc0 = MM_FMADD(b, a0, acc0);
        acc1 = MM_FMADD(b, a1, acc1);

        fp_vector_t c0 = MM_MUL_Px(MM_MASKLOADU_Px(&cc[0],   mask_n), beta_ps);
        fp_vector_t c1 = MM_MUL_Px(MM_MASKLOADU_Px(&cc[ldc], mask_n), beta_ps);
        MM_MASKSTOREU_Px(&cc[0]  , mask_n, MM_FMADD(acc0, alpha_ps, c0));
        MM_MASKSTOREU_Px(&cc[ldc], mask_n, MM_FMADD(acc1, alpha_ps, c1));
      }
      A += lda*2;
      C += ldc*2;
    }
    if ((unsigned)pPrm->M % 2) {
      const scalar_t *bb = pPrm->B;
      scalar_t *cc = C;
      for (unsigned ni = nw; ni != 0; bb += SIMD_FACTOR, cc += SIMD_FACTOR, --ni) {
        fp_vector_t a0, b;

        a0 = MM_BROADCAST_Sx(&A[    0]);
        b = MM_LOADU_Px(&bb[0]);
        fp_vector_t acc0 = MM_MUL_Px(b, a0);

        a0 = MM_BROADCAST_Sx(&A[    1]);
        b = MM_LOADU_Px(&bb[ldb1]);
        acc0 = MM_FMADD(b, a0, acc0);

        a0 = MM_BROADCAST_Sx(&A[    2]);
        b = MM_LOADU_Px(&bb[ldb2]);
        acc0 = MM_FMADD(b, a0, acc0);

        a0 = MM_BROADCAST_Sx(&A[    3]);
        b = MM_LOADU_Px(&bb[ldb3]);
        acc0 = MM_FMADD(b, a0, acc0);

        a0 = MM_BROADCAST_Sx(&A[    4]);
        b = MM_LOADU_Px(&bb[ldb4]);
        acc0 = MM_FMADD(b, a0, acc0);

        fp_vector_t c0 = MM_MUL_Px(MM_LOADU_Px(&cc[0])  , beta_ps);
        MM_STOREU_Px(&cc[0]  , MM_FMADD(acc0, alpha_ps, c0));
      }
      if (nRem) {
        fp_vector_t a0, b;
        int_vector_t mask_n = pPrm->mask_n;

        a0 = MM_BROADCAST_Sx(&A[    0]);
        b = MM_MASKLOADU_Px(&bb[0], mask_n);
        fp_vector_t acc0 = MM_MUL_Px(b, a0);

        a0 = MM_BROADCAST_Sx(&A[    1]);
        b = MM_MASKLOADU_Px(&bb[ldb1], mask_n);
        acc0 = MM_FMADD(b, a0, acc0);

        a0 = MM_BROADCAST_Sx(&A[    2]);
        b = MM_MASKLOADU_Px(&bb[ldb2], mask_n);
        acc0 = MM_FMADD(b, a0, acc0);

        a0 = MM_BROADCAST_Sx(&A[    3]);
        b = MM_MASKLOADU_Px(&bb[ldb3], mask_n);
        acc0 = MM_FMADD(b, a0, acc0);

        a0 = MM_BROADCAST_Sx(&A[    4]);
        b = MM_MASKLOADU_Px(&bb[ldb4], mask_n);
        acc0 = MM_FMADD(b, a0, acc0);

        fp_vector_t c0 = MM_MUL_Px(MM_MASKLOADU_Px(&cc[0],   mask_n), beta_ps);
        MM_MASKSTOREU_Px(&cc[0]  , mask_n, MM_FMADD(acc0, alpha_ps, c0));
      }
    }
  } else if (K == 4) {
    const int ldb1 = pPrm->ldb;
    const int ldb2 = ldb1+ldb1;
    const int ldb3 = ldb2+ldb1;
    for (unsigned mh = (unsigned)pPrm->M / 2; mh != 0; --mh) {
      const scalar_t *bb = pPrm->B;
      scalar_t *cc = C;
      fp_vector_t a00 = MM_BROADCAST_Sx(&A[0]);
      fp_vector_t a01 = MM_BROADCAST_Sx(&A[1]);
      fp_vector_t a02 = MM_BROADCAST_Sx(&A[2]);
      fp_vector_t a03 = MM_BROADCAST_Sx(&A[3]);
      A += lda;
      fp_vector_t a10 = MM_BROADCAST_Sx(&A[0]);
      fp_vector_t a11 = MM_BROADCAST_Sx(&A[1]);
      fp_vector_t a12 = MM_BROADCAST_Sx(&A[2]);
      fp_vector_t a13 = MM_BROADCAST_Sx(&A[3]);
      A += lda;
      for (unsigned ni = nw; ni != 0; bb += SIMD_FACTOR, cc += SIMD_FACTOR, --ni) {
        fp_vector_t b;

        b = MM_LOADU_Px(&bb[0]);
        fp_vector_t acc0 = MM_MUL_Px(b, a00);
        fp_vector_t acc1 = MM_MUL_Px(b, a10);

        b = MM_LOADU_Px(&bb[ldb1]);
        acc0 = MM_FMADD(b, a01, acc0);
        acc1 = MM_FMADD(b, a11, acc1);

        b = MM_LOADU_Px(&bb[ldb2]);
        acc0 = MM_FMADD(b, a02, acc0);
        acc1 = MM_FMADD(b, a12, acc1);

        b = MM_LOADU_Px(&bb[ldb3]);
        acc0 = MM_FMADD(b, a03, acc0);
        acc1 = MM_FMADD(b, a13, acc1);

        fp_vector_t c0 = MM_MUL_Px(MM_LOADU_Px(&cc[0])  , beta_ps);
        fp_vector_t c1 = MM_MUL_Px(MM_LOADU_Px(&cc[ldc]), beta_ps);
        MM_STOREU_Px(&cc[0]  , MM_FMADD(acc0, alpha_ps, c0));
        MM_STOREU_Px(&cc[ldc], MM_FMADD(acc1, alpha_ps, c1));
      }
      if (nRem) {
        fp_vector_t b;
        int_vector_t mask_n = pPrm->mask_n;

        b = MM_MASKLOADU_Px(&bb[0], mask_n);
        fp_vector_t acc0 = MM_MUL_Px(b, a00);
        fp_vector_t acc1 = MM_MUL_Px(b, a10);

        b = MM_MASKLOADU_Px(&bb[ldb1], mask_n);
        acc0 = MM_FMADD(b, a01, acc0);
        acc1 = MM_FMADD(b, a11, acc1);

        b = MM_MASKLOADU_Px(&bb[ldb2], mask_n);
        acc0 = MM_FMADD(b, a02, acc0);
        acc1 = MM_FMADD(b, a12, acc1);

        b = MM_MASKLOADU_Px(&bb[ldb3], mask_n);
        acc0 = MM_FMADD(b, a03, acc0);
        acc1 = MM_FMADD(b, a13, acc1);

        fp_vector_t c0 = MM_MUL_Px(MM_MASKLOADU_Px(&cc[0],   mask_n), beta_ps);
        fp_vector_t c1 = MM_MUL_Px(MM_MASKLOADU_Px(&cc[ldc], mask_n), beta_ps);
        MM_MASKSTOREU_Px(&cc[0]  , mask_n, MM_FMADD(acc0, alpha_ps, c0));
        MM_MASKSTOREU_Px(&cc[ldc], mask_n, MM_FMADD(acc1, alpha_ps, c1));
      }
      C += ldc*2;
    }
    if ((unsigned)pPrm->M % 2) {
      const scalar_t *bb = pPrm->B;
      scalar_t *cc = C;
      fp_vector_t a00 = MM_BROADCAST_Sx(&A[0]);
      fp_vector_t a01 = MM_BROADCAST_Sx(&A[1]);
      fp_vector_t a02 = MM_BROADCAST_Sx(&A[2]);
      fp_vector_t a03 = MM_BROADCAST_Sx(&A[3]);
      for (unsigned ni = nw; ni != 0; bb += SIMD_FACTOR, cc += SIMD_FACTOR, --ni) {
        fp_vector_t b;

        b = MM_LOADU_Px(&bb[0]);
        fp_vector_t acc0 = MM_MUL_Px(b, a00);

        b = MM_LOADU_Px(&bb[ldb1]);
        acc0 = MM_FMADD(b, a01, acc0);

        b = MM_LOADU_Px(&bb[ldb2]);
        acc0 = MM_FMADD(b, a02, acc0);

        b = MM_LOADU_Px(&bb[ldb3]);
        acc0 = MM_FMADD(b, a03, acc0);

        fp_vector_t c0 = MM_MUL_Px(MM_LOADU_Px(&cc[0])  , beta_ps);
        MM_STOREU_Px(&cc[0]  , MM_FMADD(acc0, alpha_ps, c0));
      }
      if (nRem) {
        fp_vector_t b;
        int_vector_t mask_n = pPrm->mask_n;

        b = MM_MASKLOADU_Px(&bb[0], mask_n);
        fp_vector_t acc0 = MM_MUL_Px(b, a00);

        b = MM_MASKLOADU_Px(&bb[ldb1], mask_n);
        acc0 = MM_FMADD(b, a01, acc0);

        b = MM_MASKLOADU_Px(&bb[ldb2], mask_n);
        acc0 = MM_FMADD(b, a02, acc0);

        b = MM_MASKLOADU_Px(&bb[ldb3], mask_n);
        acc0 = MM_FMADD(b, a03, acc0);

        fp_vector_t c0 = MM_MUL_Px(MM_MASKLOADU_Px(&cc[0],   mask_n), beta_ps);
        MM_MASKSTOREU_Px(&cc[0]  , mask_n, MM_FMADD(acc0, alpha_ps, c0));
      }
    }
  } else if (K == 3) {
    const int ldb1 = pPrm->ldb;
    const int ldb2 = ldb1+ldb1;
    for (unsigned mh = (unsigned)pPrm->M / 2; mh != 0; --mh) {
      const scalar_t *bb = pPrm->B;
      scalar_t *cc = C;
      fp_vector_t a00 = MM_BROADCAST_Sx(&A[0]);
      fp_vector_t a01 = MM_BROADCAST_Sx(&A[1]);
      fp_vector_t a02 = MM_BROADCAST_Sx(&A[2]);
      A += lda;
      fp_vector_t a10 = MM_BROADCAST_Sx(&A[0]);
      fp_vector_t a11 = MM_BROADCAST_Sx(&A[1]);
      fp_vector_t a12 = MM_BROADCAST_Sx(&A[2]);
      A += lda;
      for (unsigned ni = nw; ni != 0; bb += SIMD_FACTOR, cc += SIMD_FACTOR, --ni) {
        fp_vector_t b;

        b = MM_LOADU_Px(&bb[0]);
        fp_vector_t acc0 = MM_MUL_Px(b, a00);
        fp_vector_t acc1 = MM_MUL_Px(b, a10);

        b = MM_LOADU_Px(&bb[ldb1]);
        acc0 = MM_FMADD(b, a01, acc0);
        acc1 = MM_FMADD(b, a11, acc1);

        b = MM_LOADU_Px(&bb[ldb2]);
        acc0 = MM_FMADD(b, a02, acc0);
        acc1 = MM_FMADD(b, a12, acc1);

        fp_vector_t c0 = MM_MUL_Px(MM_LOADU_Px(&cc[0])  , beta_ps);
        fp_vector_t c1 = MM_MUL_Px(MM_LOADU_Px(&cc[ldc]), beta_ps);
        MM_STOREU_Px(&cc[0]  , MM_FMADD(acc0, alpha_ps, c0));
        MM_STOREU_Px(&cc[ldc], MM_FMADD(acc1, alpha_ps, c1));
      }
      if (nRem) {
        fp_vector_t b;
        int_vector_t mask_n = pPrm->mask_n;

        b = MM_MASKLOADU_Px(&bb[0], mask_n);
        fp_vector_t acc0 = MM_MUL_Px(b, a00);
        fp_vector_t acc1 = MM_MUL_Px(b, a10);

        b = MM_MASKLOADU_Px(&bb[ldb1], mask_n);
        acc0 = MM_FMADD(b, a01, acc0);
        acc1 = MM_FMADD(b, a11, acc1);

        b = MM_MASKLOADU_Px(&bb[ldb2], mask_n);
        acc0 = MM_FMADD(b, a02, acc0);
        acc1 = MM_FMADD(b, a12, acc1);

        fp_vector_t c0 = MM_MUL_Px(MM_MASKLOADU_Px(&cc[0],   mask_n), beta_ps);
        fp_vector_t c1 = MM_MUL_Px(MM_MASKLOADU_Px(&cc[ldc], mask_n), beta_ps);
        MM_MASKSTOREU_Px(&cc[0]  , mask_n, MM_FMADD(acc0, alpha_ps, c0));
        MM_MASKSTOREU_Px(&cc[ldc], mask_n, MM_FMADD(acc1, alpha_ps, c1));
      }
      C += ldc*2;
    }
    if ((unsigned)pPrm->M % 2) {
      const scalar_t *bb = pPrm->B;
      scalar_t *cc = C;
      fp_vector_t a00 = MM_BROADCAST_Sx(&A[0]);
      fp_vector_t a01 = MM_BROADCAST_Sx(&A[1]);
      fp_vector_t a02 = MM_BROADCAST_Sx(&A[2]);
      for (unsigned ni = nw; ni != 0; bb += SIMD_FACTOR, cc += SIMD_FACTOR, --ni) {
        fp_vector_t b;

        b = MM_LOADU_Px(&bb[0]);
        fp_vector_t acc0 = MM_MUL_Px(b, a00);

        b = MM_LOADU_Px(&bb[ldb1]);
        acc0 = MM_FMADD(b, a01, acc0);

        b = MM_LOADU_Px(&bb[ldb2]);
        acc0 = MM_FMADD(b, a02, acc0);

        fp_vector_t c0 = MM_MUL_Px(MM_LOADU_Px(&cc[0])  , beta_ps);
        MM_STOREU_Px(&cc[0]  , MM_FMADD(acc0, alpha_ps, c0));
      }
      if (nRem) {
        fp_vector_t b;
        int_vector_t mask_n = pPrm->mask_n;

        b = MM_MASKLOADU_Px(&bb[0], mask_n);
        fp_vector_t acc0 = MM_MUL_Px(b, a00);

        b = MM_MASKLOADU_Px(&bb[ldb1], mask_n);
        acc0 = MM_FMADD(b, a01, acc0);

        b = MM_MASKLOADU_Px(&bb[ldb2], mask_n);
        acc0 = MM_FMADD(b, a02, acc0);

        fp_vector_t c0 = MM_MUL_Px(MM_MASKLOADU_Px(&cc[0],   mask_n), beta_ps);
        MM_MASKSTOREU_Px(&cc[0]  , mask_n, MM_FMADD(acc0, alpha_ps, c0));
      }
    }
  } else if (K == 2) {
    const int ldb1 = pPrm->ldb;
    for (unsigned mh = (unsigned)pPrm->M / 2; mh != 0; --mh) {
      const scalar_t *bb = pPrm->B;
      scalar_t *cc = C;
      fp_vector_t a00 = MM_BROADCAST_Sx(&A[0]);
      fp_vector_t a01 = MM_BROADCAST_Sx(&A[1]);
      A += lda;
      fp_vector_t a10 = MM_BROADCAST_Sx(&A[0]);
      fp_vector_t a11 = MM_BROADCAST_Sx(&A[1]);
      A += lda;
      for (unsigned ni = nw; ni != 0; bb += SIMD_FACTOR, cc += SIMD_FACTOR, --ni) {
        fp_vector_t b;

        b = MM_LOADU_Px(&bb[0]);
        fp_vector_t acc0 = MM_MUL_Px(b, a00);
        fp_vector_t acc1 = MM_MUL_Px(b, a10);

        b = MM_LOADU_Px(&bb[ldb1]);
        acc0 = MM_FMADD(b, a01, acc0);
        acc1 = MM_FMADD(b, a11, acc1);

        fp_vector_t c0 = MM_MUL_Px(MM_LOADU_Px(&cc[0])  , beta_ps);
        fp_vector_t c1 = MM_MUL_Px(MM_LOADU_Px(&cc[ldc]), beta_ps);
        MM_STOREU_Px(&cc[0]  , MM_FMADD(acc0, alpha_ps, c0));
        MM_STOREU_Px(&cc[ldc], MM_FMADD(acc1, alpha_ps, c1));
      }
      if (nRem) {
        fp_vector_t b;
        int_vector_t mask_n = pPrm->mask_n;

        b = MM_MASKLOADU_Px(&bb[0], mask_n);
        fp_vector_t acc0 = MM_MUL_Px(b, a00);
        fp_vector_t acc1 = MM_MUL_Px(b, a10);

        b = MM_MASKLOADU_Px(&bb[ldb1], mask_n);
        acc0 = MM_FMADD(b, a01, acc0);
        acc1 = MM_FMADD(b, a11, acc1);

        fp_vector_t c0 = MM_MUL_Px(MM_MASKLOADU_Px(&cc[0],   mask_n), beta_ps);
        fp_vector_t c1 = MM_MUL_Px(MM_MASKLOADU_Px(&cc[ldc], mask_n), beta_ps);
        MM_MASKSTOREU_Px(&cc[0]  , mask_n, MM_FMADD(acc0, alpha_ps, c0));
        MM_MASKSTOREU_Px(&cc[ldc], mask_n, MM_FMADD(acc1, alpha_ps, c1));
      }
      C += ldc*2;
    }
    if ((unsigned)pPrm->M % 2) {
      const scalar_t *bb = pPrm->B;
      scalar_t *cc = C;
      fp_vector_t a00 = MM_BROADCAST_Sx(&A[0]);
      fp_vector_t a01 = MM_BROADCAST_Sx(&A[1]);
      for (unsigned ni = nw; ni != 0; bb += SIMD_FACTOR, cc += SIMD_FACTOR, --ni) {
        fp_vector_t b;

        b = MM_LOADU_Px(&bb[0]);
        fp_vector_t acc0 = MM_MUL_Px(b, a00);

        b = MM_LOADU_Px(&bb[ldb1]);
        acc0 = MM_FMADD(b, a01, acc0);

        fp_vector_t c0 = MM_MUL_Px(MM_LOADU_Px(&cc[0])  , beta_ps);
        MM_STOREU_Px(&cc[0]  , MM_FMADD(acc0, alpha_ps, c0));
      }
      if (nRem) {
        fp_vector_t b;
        int_vector_t mask_n = pPrm->mask_n;

        b = MM_MASKLOADU_Px(&bb[0], mask_n);
        fp_vector_t acc0 = MM_MUL_Px(b, a00);

        b = MM_MASKLOADU_Px(&bb[ldb1], mask_n);
        acc0 = MM_FMADD(b, a01, acc0);

        fp_vector_t c0 = MM_MUL_Px(MM_MASKLOADU_Px(&cc[0],   mask_n), beta_ps);
        MM_MASKSTOREU_Px(&cc[0]  , mask_n, MM_FMADD(acc0, alpha_ps, c0));
      }
    }
  } else { // K==1
    for (unsigned mh = (unsigned)pPrm->M / 2; mh != 0; --mh) {
      const scalar_t *bb = pPrm->B;
      scalar_t *cc = C;
      fp_vector_t a00 = MM_BROADCAST_Sx(&A[0]);
      A += lda;
      fp_vector_t a10 = MM_BROADCAST_Sx(&A[0]);
      A += lda;
      for (unsigned ni = nw; ni != 0; bb += SIMD_FACTOR, cc += SIMD_FACTOR, --ni) {
        fp_vector_t b;

        b = MM_LOADU_Px(&bb[0]);
        fp_vector_t acc0 = MM_MUL_Px(b, a00);
        fp_vector_t acc1 = MM_MUL_Px(b, a10);

        fp_vector_t c0 = MM_MUL_Px(MM_LOADU_Px(&cc[0])  , beta_ps);
        fp_vector_t c1 = MM_MUL_Px(MM_LOADU_Px(&cc[ldc]), beta_ps);
        MM_STOREU_Px(&cc[0]  , MM_FMADD(acc0, alpha_ps, c0));
        MM_STOREU_Px(&cc[ldc], MM_FMADD(acc1, alpha_ps, c1));
      }
      if (nRem) {
        fp_vector_t b;
        int_vector_t mask_n = pPrm->mask_n;

        b = MM_MASKLOADU_Px(&bb[0], mask_n);
        fp_vector_t acc0 = MM_MUL_Px(b, a00);
        fp_vector_t acc1 = MM_MUL_Px(b, a10);

        fp_vector_t c0 = MM_MUL_Px(MM_MASKLOADU_Px(&cc[0],   mask_n), beta_ps);
        fp_vector_t c1 = MM_MUL_Px(MM_MASKLOADU_Px(&cc[ldc], mask_n), beta_ps);
        MM_MASKSTOREU_Px(&cc[0]  , mask_n, MM_FMADD(acc0, alpha_ps, c0));
        MM_MASKSTOREU_Px(&cc[ldc], mask_n, MM_FMADD(acc1, alpha_ps, c1));
      }
      C += ldc*2;
    }
    if ((unsigned)pPrm->M % 2) {
      const scalar_t *bb = pPrm->B;
      scalar_t *cc = C;
      fp_vector_t a00 = MM_BROADCAST_Sx(&A[0]);
      for (unsigned ni = nw; ni != 0; bb += SIMD_FACTOR, cc += SIMD_FACTOR, --ni) {
        fp_vector_t b;

        b = MM_LOADU_Px(&bb[0]);
        fp_vector_t acc0 = MM_MUL_Px(b, a00);

        fp_vector_t c0 = MM_MUL_Px(MM_LOADU_Px(&cc[0])  , beta_ps);
        MM_STOREU_Px(&cc[0]  , MM_FMADD(acc0, alpha_ps, c0));
      }
      if (nRem) {
        fp_vector_t b;
        int_vector_t mask_n = pPrm->mask_n;

        b = MM_MASKLOADU_Px(&bb[0], mask_n);
        fp_vector_t acc0 = MM_MUL_Px(b, a00);

        fp_vector_t c0 = MM_MUL_Px(MM_MASKLOADU_Px(&cc[0],   mask_n), beta_ps);
        MM_MASKSTOREU_Px(&cc[0]  , mask_n, MM_FMADD(acc0, alpha_ps, c0));
      }
    }
  }
}

static char* my_aligned_alloc(int size, char** pFreePtr)
{
  char* p = malloc(size+sizeof(fp_vector_t));
  if (p == 0)
    return p;
  *pFreePtr = p;
  uintptr_t adj = (0-(uintptr_t)(p)) % sizeof(fp_vector_t);
  return p + adj;
}

// K >= 8
// N <= 4*SIMD_FACTOR
static void avx256_noncblas_sgemm_smallN(gemm_prm_t* pPrm)
{
  int nw = (unsigned)(pPrm->N-1) / SIMD_FACTOR;
  const scalar_t *B = pPrm->B;
  const int ldb     = pPrm->ldb;
  if (pPrm->M > 5 &&
      pPrm->K > 5 &&
    (((uintptr_t)B % sizeof(fp_vector_t) != 0) || (ldb % SIMD_FACTOR != 0))) {
    // copy B into aligned buffer
    char* bbufFreePtr = 0;
    fp_vector_t* bbuf = (fp_vector_t*)my_aligned_alloc((nw+1)*pPrm->K*sizeof(fp_vector_t), &bbufFreePtr); // hopefully K is not too big
    if (bbuf) {
      if (nw == 3) {
        fp_vector_t* bdst = bbuf;
        const scalar_t* bsrc = B;
        int k = pPrm->K;
        int_vector_t mask_n = pPrm->mask_n;
        do {
          bdst[0] = MM_LOADU_Px    (&bsrc[SIMD_FACTOR*0]);
          bdst[1] = MM_LOADU_Px    (&bsrc[SIMD_FACTOR*1]);
          bdst[2] = MM_LOADU_Px    (&bsrc[SIMD_FACTOR*2]);
          bdst[3] = MM_MASKLOADU_Px(&bsrc[SIMD_FACTOR*3], mask_n);
          bsrc += ldb;
          bdst += 4;
        } while (--k);
        // copy done

        int betaZ         = pPrm->beta==0;
        const scalar_t *A = pPrm->A;
        scalar_t *C       = pPrm->C;
        const int lda     = pPrm->lda;
        const int ldc     = pPrm->ldc;
        const int bstep   = 4;
        const unsigned kRem = (unsigned)pPrm->K % 4;
        const unsigned kDiv = ((unsigned)pPrm->K-1) / 4;
        for (unsigned mh = (unsigned)pPrm->M / 2; mh != 0; --mh) {
          const scalar_t *aa = A;
          const fp_vector_t *bb = bbuf;
          fp_vector_t b;
          fp_vector_t a0 = MM_BROADCAST_Sx(&aa[    0]);
          fp_vector_t a1 = MM_BROADCAST_Sx(&aa[lda+0]);
          aa += 1;

          b = bb[0];
          fp_vector_t acc00 = MM_MUL_Px(b, a0);
          fp_vector_t acc10 = MM_MUL_Px(b, a1);

          b = bb[1];
          fp_vector_t acc01 = MM_MUL_Px(b, a0);
          fp_vector_t acc11 = MM_MUL_Px(b, a1);

          b = bb[2];
          fp_vector_t acc02 = MM_MUL_Px(b, a0);
          fp_vector_t acc12 = MM_MUL_Px(b, a1);

          b = bb[3];
          fp_vector_t acc03 = MM_MUL_Px(b, a0);
          fp_vector_t acc13 = MM_MUL_Px(b, a1);
          bb += bstep;

          if (kRem != 1) {
            a0 = MM_BROADCAST_Sx(&aa[    0]);
            a1 = MM_BROADCAST_Sx(&aa[lda+0]);
            aa += 1;

            b = bb[0];
            acc00 = MM_FMADD(b, a0, acc00);
            acc10 = MM_FMADD(b, a1, acc10);

            b = bb[1];
            acc01 = MM_FMADD(b, a0, acc01);
            acc11 = MM_FMADD(b, a1, acc11);

            b = bb[2];
            acc02 = MM_FMADD(b, a0, acc02);
            acc12 = MM_FMADD(b, a1, acc12);

            b = bb[3];
            acc03 = MM_FMADD(b, a0, acc03);
            acc13 = MM_FMADD(b, a1, acc13);

            bb += bstep;
            if (kRem != 2) {
              a0 = MM_BROADCAST_Sx(&aa[    0]);
              a1 = MM_BROADCAST_Sx(&aa[lda+0]);
              aa += 1;

              b = bb[0];
              acc00 = MM_FMADD(b, a0, acc00);
              acc10 = MM_FMADD(b, a1, acc10);

              b = bb[1];
              acc01 = MM_FMADD(b, a0, acc01);
              acc11 = MM_FMADD(b, a1, acc11);

              b = bb[2];
              acc02 = MM_FMADD(b, a0, acc02);
              acc12 = MM_FMADD(b, a1, acc12);

              b = bb[3];
              acc03 = MM_FMADD(b, a0, acc03);
              acc13 = MM_FMADD(b, a1, acc13);

              bb += bstep;
              if (kRem != 3) {
                a0 = MM_BROADCAST_Sx(&aa[    0]);
                a1 = MM_BROADCAST_Sx(&aa[lda+0]);
                aa += 1;

                b = bb[0];
                acc00 = MM_FMADD(b, a0, acc00);
                acc10 = MM_FMADD(b, a1, acc10);

                b = bb[1];
                acc01 = MM_FMADD(b, a0, acc01);
                acc11 = MM_FMADD(b, a1, acc11);

                b = bb[2];
                acc02 = MM_FMADD(b, a0, acc02);
                acc12 = MM_FMADD(b, a1, acc12);

                b = bb[3];
                acc03 = MM_FMADD(b, a0, acc03);
                acc13 = MM_FMADD(b, a1, acc13);

                bb += bstep;
              }
            }
          }

          int k = kDiv;
          do {
            a0 = MM_BROADCAST_Sx(&aa[    0]);
            a1 = MM_BROADCAST_Sx(&aa[lda+0]);
            aa += 1;

            b = bb[0];
            acc00 = MM_FMADD(b, a0, acc00);
            acc10 = MM_FMADD(b, a1, acc10);

            b = bb[1];
            acc01 = MM_FMADD(b, a0, acc01);
            acc11 = MM_FMADD(b, a1, acc11);

            b = bb[2];
            acc02 = MM_FMADD(b, a0, acc02);
            acc12 = MM_FMADD(b, a1, acc12);

            b = bb[3];
            acc03 = MM_FMADD(b, a0, acc03);
            acc13 = MM_FMADD(b, a1, acc13);

            bb += bstep;

            a0 = MM_BROADCAST_Sx(&aa[    0]);
            a1 = MM_BROADCAST_Sx(&aa[lda+0]);
            aa += 1;

            b = bb[0];
            acc00 = MM_FMADD(b, a0, acc00);
            acc10 = MM_FMADD(b, a1, acc10);

            b = bb[1];
            acc01 = MM_FMADD(b, a0, acc01);
            acc11 = MM_FMADD(b, a1, acc11);

            b = bb[2];
            acc02 = MM_FMADD(b, a0, acc02);
            acc12 = MM_FMADD(b, a1, acc12);

            b = bb[3];
            acc03 = MM_FMADD(b, a0, acc03);
            acc13 = MM_FMADD(b, a1, acc13);

            bb += bstep;

            a0 = MM_BROADCAST_Sx(&aa[    0]);
            a1 = MM_BROADCAST_Sx(&aa[lda+0]);
            aa += 1;

            b = bb[0];
            acc00 = MM_FMADD(b, a0, acc00);
            acc10 = MM_FMADD(b, a1, acc10);

            b = bb[1];
            acc01 = MM_FMADD(b, a0, acc01);
            acc11 = MM_FMADD(b, a1, acc11);

            b = bb[2];
            acc02 = MM_FMADD(b, a0, acc02);
            acc12 = MM_FMADD(b, a1, acc12);

            b = bb[3];
            acc03 = MM_FMADD(b, a0, acc03);
            acc13 = MM_FMADD(b, a1, acc13);

            bb += bstep;

            a0 = MM_BROADCAST_Sx(&aa[    0]);
            a1 = MM_BROADCAST_Sx(&aa[lda+0]);
            aa += 1;

            b = bb[0];
            acc00 = MM_FMADD(b, a0, acc00);
            acc10 = MM_FMADD(b, a1, acc10);

            b = bb[1];
            acc01 = MM_FMADD(b, a0, acc01);
            acc11 = MM_FMADD(b, a1, acc11);

            b = bb[2];
            acc02 = MM_FMADD(b, a0, acc02);
            acc12 = MM_FMADD(b, a1, acc12);

            b = bb[3];
            acc03 = MM_FMADD(b, a0, acc03);
            acc13 = MM_FMADD(b, a1, acc13);

            bb += bstep;
          } while (--k);

          fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
          if (betaZ) {
            MM_STOREU_Px    (&C[SIMD_FACTOR*0],         MM_MUL_Px(acc00, alpha_ps));
            MM_STOREU_Px    (&C[SIMD_FACTOR*1],         MM_MUL_Px(acc01, alpha_ps));
            MM_STOREU_Px    (&C[SIMD_FACTOR*2],         MM_MUL_Px(acc02, alpha_ps));
            MM_MASKSTOREU_Px(&C[SIMD_FACTOR*3], mask_n, MM_MUL_Px(acc03, alpha_ps));
            C += ldc;

            MM_STOREU_Px    (&C[SIMD_FACTOR*0],         MM_MUL_Px(acc10, alpha_ps));
            MM_STOREU_Px    (&C[SIMD_FACTOR*1],         MM_MUL_Px(acc11, alpha_ps));
            MM_STOREU_Px    (&C[SIMD_FACTOR*2],         MM_MUL_Px(acc12, alpha_ps));
            MM_MASKSTOREU_Px(&C[SIMD_FACTOR*3], mask_n, MM_MUL_Px(acc13, alpha_ps));
            C += ldc;
          } else {
            fp_vector_t beta_ps  = MM_BROADCAST_Sx(&pPrm->beta);

            MM_STOREU_Px    (&C[SIMD_FACTOR*0],         MM_FMADD(acc00, alpha_ps, MM_MUL_Px(MM_LOADU_Px    (&C[SIMD_FACTOR*0]), beta_ps)));
            MM_STOREU_Px    (&C[SIMD_FACTOR*1],         MM_FMADD(acc01, alpha_ps, MM_MUL_Px(MM_LOADU_Px    (&C[SIMD_FACTOR*1]), beta_ps)));
            MM_STOREU_Px    (&C[SIMD_FACTOR*2],         MM_FMADD(acc02, alpha_ps, MM_MUL_Px(MM_LOADU_Px    (&C[SIMD_FACTOR*2]), beta_ps)));
            MM_MASKSTOREU_Px(&C[SIMD_FACTOR*3], mask_n, MM_FMADD(acc03, alpha_ps, MM_MUL_Px(MM_MASKLOADU_Px(&C[SIMD_FACTOR*3], mask_n), beta_ps)));
            C += ldc;

            MM_STOREU_Px    (&C[SIMD_FACTOR*0],         MM_FMADD(acc10, alpha_ps, MM_MUL_Px(MM_LOADU_Px    (&C[SIMD_FACTOR*0]), beta_ps)));
            MM_STOREU_Px    (&C[SIMD_FACTOR*1],         MM_FMADD(acc11, alpha_ps, MM_MUL_Px(MM_LOADU_Px    (&C[SIMD_FACTOR*1]), beta_ps)));
            MM_STOREU_Px    (&C[SIMD_FACTOR*2],         MM_FMADD(acc12, alpha_ps, MM_MUL_Px(MM_LOADU_Px    (&C[SIMD_FACTOR*2]), beta_ps)));
            MM_MASKSTOREU_Px(&C[SIMD_FACTOR*3], mask_n, MM_FMADD(acc13, alpha_ps, MM_MUL_Px(MM_MASKLOADU_Px(&C[SIMD_FACTOR*3], mask_n), beta_ps)));
            C += ldc;
          }

          A += lda*2;
        }
        if ((unsigned)pPrm->M % 2) {
        }
      } else if (nw==2) {
      } else if (nw==1) {
      } else {  // (nw==0)
      }
      free(bbufFreePtr);
    }
  } else {
    int betaZ = pPrm->beta==0;
    const scalar_t *A = pPrm->A;
    scalar_t *C       = pPrm->C;
    const int lda     = pPrm->lda;
    const int ldc     = pPrm->ldc;
    int_vector_t mask_n = pPrm->mask_n;
    if (nw == 3) {
      for (unsigned mh = (unsigned)pPrm->M / 2; mh != 0; --mh) {
        const scalar_t *aa = A;
        const scalar_t *bb = B;
        fp_vector_t b;
        fp_vector_t a0 = MM_BROADCAST_Sx(&aa[    0]);
        fp_vector_t a1 = MM_BROADCAST_Sx(&aa[lda+0]);
        aa += 1;

        b = MM_LOADU_Px(&bb[SIMD_FACTOR*0]);
        fp_vector_t acc00 = MM_MUL_Px(b, a0);
        fp_vector_t acc10 = MM_MUL_Px(b, a1);

        b = MM_LOADU_Px(&bb[SIMD_FACTOR*1]);
        fp_vector_t acc01 = MM_MUL_Px(b, a0);
        fp_vector_t acc11 = MM_MUL_Px(b, a1);

        b = MM_LOADU_Px(&bb[SIMD_FACTOR*2]);
        fp_vector_t acc02 = MM_MUL_Px(b, a0);
        fp_vector_t acc12 = MM_MUL_Px(b, a1);

        b = MM_MASKLOADU_Px(&bb[SIMD_FACTOR*3], mask_n);
        fp_vector_t acc03 = MM_MUL_Px(b, a0);
        fp_vector_t acc13 = MM_MUL_Px(b, a1);
        bb += ldb;

        if ((unsigned)pPrm->K % 2 == 0) {
          a0 = MM_BROADCAST_Sx(&aa[    0]);
          a1 = MM_BROADCAST_Sx(&aa[lda+0]);
          aa += 1;

          b = MM_LOADU_Px(&bb[SIMD_FACTOR*0]);
          acc00 = MM_FMADD(b, a0, acc00);
          acc10 = MM_FMADD(b, a1, acc10);

          b = MM_LOADU_Px(&bb[SIMD_FACTOR*1]);
          acc01 = MM_FMADD(b, a0, acc01);
          acc11 = MM_FMADD(b, a1, acc11);

          b = MM_LOADU_Px(&bb[SIMD_FACTOR*2]);
          acc02 = MM_FMADD(b, a0, acc02);
          acc12 = MM_FMADD(b, a1, acc12);

          b = MM_MASKLOADU_Px(&bb[SIMD_FACTOR*3], mask_n);
          acc03 = MM_FMADD(b, a0, acc03);
          acc13 = MM_FMADD(b, a1, acc13);

          bb += ldb;
        }

        int k = ((unsigned)pPrm->K-1) / 2;
        do {
          a0 = MM_BROADCAST_Sx(&aa[    0]);
          a1 = MM_BROADCAST_Sx(&aa[lda+0]);
          aa += 1;

          b = MM_LOADU_Px(&bb[SIMD_FACTOR*0]);
          acc00 = MM_FMADD(b, a0, acc00);
          acc10 = MM_FMADD(b, a1, acc10);

          b = MM_LOADU_Px(&bb[SIMD_FACTOR*1]);
          acc01 = MM_FMADD(b, a0, acc01);
          acc11 = MM_FMADD(b, a1, acc11);

          b = MM_LOADU_Px(&bb[SIMD_FACTOR*2]);
          acc02 = MM_FMADD(b, a0, acc02);
          acc12 = MM_FMADD(b, a1, acc12);

          b = MM_MASKLOADU_Px(&bb[SIMD_FACTOR*3], mask_n);
          acc03 = MM_FMADD(b, a0, acc03);
          acc13 = MM_FMADD(b, a1, acc13);

          bb += ldb;

          a0 = MM_BROADCAST_Sx(&aa[    0]);
          a1 = MM_BROADCAST_Sx(&aa[lda+0]);
          aa += 1;

          b = MM_LOADU_Px(&bb[SIMD_FACTOR*0]);
          acc00 = MM_FMADD(b, a0, acc00);
          acc10 = MM_FMADD(b, a1, acc10);

          b = MM_LOADU_Px(&bb[SIMD_FACTOR*1]);
          acc01 = MM_FMADD(b, a0, acc01);
          acc11 = MM_FMADD(b, a1, acc11);

          b = MM_LOADU_Px(&bb[SIMD_FACTOR*2]);
          acc02 = MM_FMADD(b, a0, acc02);
          acc12 = MM_FMADD(b, a1, acc12);

          b = MM_MASKLOADU_Px(&bb[SIMD_FACTOR*3], mask_n);
          acc03 = MM_FMADD(b, a0, acc03);
          acc13 = MM_FMADD(b, a1, acc13);

          bb += ldb;
        } while (--k);

        fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
        if (betaZ) {
          MM_STOREU_Px    (&C[SIMD_FACTOR*0],         MM_MUL_Px(acc00, alpha_ps));
          MM_STOREU_Px    (&C[SIMD_FACTOR*1],         MM_MUL_Px(acc01, alpha_ps));
          MM_STOREU_Px    (&C[SIMD_FACTOR*2],         MM_MUL_Px(acc02, alpha_ps));
          MM_MASKSTOREU_Px(&C[SIMD_FACTOR*3], mask_n, MM_MUL_Px(acc03, alpha_ps));
          C += ldc;

          MM_STOREU_Px    (&C[SIMD_FACTOR*0],         MM_MUL_Px(acc10, alpha_ps));
          MM_STOREU_Px    (&C[SIMD_FACTOR*1],         MM_MUL_Px(acc11, alpha_ps));
          MM_STOREU_Px    (&C[SIMD_FACTOR*2],         MM_MUL_Px(acc12, alpha_ps));
          MM_MASKSTOREU_Px(&C[SIMD_FACTOR*3], mask_n, MM_MUL_Px(acc13, alpha_ps));
          C += ldc;
        } else {
          fp_vector_t beta_ps  = MM_BROADCAST_Sx(&pPrm->beta);

          MM_STOREU_Px    (&C[SIMD_FACTOR*0],         MM_FMADD(acc00, alpha_ps, MM_MUL_Px(MM_LOADU_Px    (&C[SIMD_FACTOR*0]), beta_ps)));
          MM_STOREU_Px    (&C[SIMD_FACTOR*1],         MM_FMADD(acc01, alpha_ps, MM_MUL_Px(MM_LOADU_Px    (&C[SIMD_FACTOR*1]), beta_ps)));
          MM_STOREU_Px    (&C[SIMD_FACTOR*2],         MM_FMADD(acc02, alpha_ps, MM_MUL_Px(MM_LOADU_Px    (&C[SIMD_FACTOR*2]), beta_ps)));
          MM_MASKSTOREU_Px(&C[SIMD_FACTOR*3], mask_n, MM_FMADD(acc03, alpha_ps, MM_MUL_Px(MM_MASKLOADU_Px(&C[SIMD_FACTOR*3], mask_n), beta_ps)));
          C += ldc;

          MM_STOREU_Px    (&C[SIMD_FACTOR*0],         MM_FMADD(acc10, alpha_ps, MM_MUL_Px(MM_LOADU_Px    (&C[SIMD_FACTOR*0]), beta_ps)));
          MM_STOREU_Px    (&C[SIMD_FACTOR*1],         MM_FMADD(acc11, alpha_ps, MM_MUL_Px(MM_LOADU_Px    (&C[SIMD_FACTOR*1]), beta_ps)));
          MM_STOREU_Px    (&C[SIMD_FACTOR*2],         MM_FMADD(acc12, alpha_ps, MM_MUL_Px(MM_LOADU_Px    (&C[SIMD_FACTOR*2]), beta_ps)));
          MM_MASKSTOREU_Px(&C[SIMD_FACTOR*3], mask_n, MM_FMADD(acc13, alpha_ps, MM_MUL_Px(MM_MASKLOADU_Px(&C[SIMD_FACTOR*3], mask_n), beta_ps)));
          C += ldc;
        }

        A += lda*2;
      }
      if ((unsigned)pPrm->M % 2) {
      }
    } else if (nw==2) {
    } else if (nw==1) {
    } else {  // (nw==0)
    }
  }
}

void avx256_noncblas_sgemm_a(
  int M, int N, int K,
  scalar_t alpha,
  const scalar_t *A, int lda,
  const scalar_t *B, int ldb,
  scalar_t beta,
  scalar_t *C, int ldc)
{
  if (M <= 0 || N <= 0 || K <= 0)
    return;

  gemm_prm_t prm;
  prm.M     = M;
  prm.N     = N;
  prm.K     = K;
  prm.alpha = alpha;
  prm.A     = A;
  prm.lda   = lda;
  prm.B     = B;
  prm.ldb   = ldb;
  prm.beta  = beta;
  prm.C     = C;
  prm.ldc   = ldc;
  memset((char*)&prm.mask_n,  -1, sizeof(prm.mask_n));
  int nRem = (unsigned)N % SIMD_FACTOR;
  if (nRem > 0) { // mask on elements of rightmost SIMD word of B and C
    memset((char*)&prm.mask_n,  0, sizeof(prm.mask_n));
    memset((char*)&prm.mask_n, -1, sizeof(scalar_t)*nRem);
  }

  if (K >= 8) {
    if (N > SIMD_FACTOR*4) {
    } else {
      // cases of small N (I can't call N=32 very small, although I probably should)
      avx256_noncblas_sgemm_smallN(&prm);
    }
  } else {
    avx256_noncblas_sgemm_smallK(&prm);  // cases of very small K
  }
}
