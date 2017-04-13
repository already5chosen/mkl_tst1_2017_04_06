#include <string.h>
#include <intrin.h>

enum {
 SIMD_FACTOR         = 8,
 COLS_PER_LOOP       = 5,
 COLS_STEPS_PER_CORE = 3,
 SIMD_ELEM_PEC_COL   = COLS_PER_LOOP*COLS_STEPS_PER_CORE,
 bb_nCols            = SIMD_ELEM_PEC_COL*SIMD_FACTOR,
 bb_nRows            = 63,
 A_WORDS_PER_ITER    = 2,
};

struct noncblas_sgemm_prm_t {
  int   M;
  int   lda;
  int   ldc;
  float alpha;
  __m256 bb[SIMD_ELEM_PEC_COL*bb_nRows];
  __m256 cc[COLS_PER_LOOP*A_WORDS_PER_ITER];
};

static void fma256_noncblas_sgemm_core(
 const noncblas_sgemm_prm_t* pPrm,
 const float  *A,
 float *C)
{
  int lda = pPrm->lda;
  int ldc = pPrm->ldc;
  int m;
  for (m = 0; m < pPrm->M-1; A += lda*A_WORDS_PER_ITER, C += ldc*A_WORDS_PER_ITER, m += A_WORDS_PER_ITER) {
    float* Crow0 = C;
    for (int n = 0; n < SIMD_ELEM_PEC_COL; n += COLS_PER_LOOP) {
      const __m256 *Bcol = &pPrm->bb[n];
      __m256 a0 = _mm256_broadcast_ss(&A[0]);
      __m256 a1 = _mm256_broadcast_ss(&A[lda]);
      __m256 b;
        b = Bcol[0];
      __m256 acc00 = _mm256_mul_ps(a0, b);
      __m256 acc01 = _mm256_mul_ps(a1, b);

        b = Bcol[1];
      __m256 acc10 = _mm256_mul_ps(a0, b);
      __m256 acc11 = _mm256_mul_ps(a1, b);

        b = Bcol[2];
      __m256 acc20 = _mm256_mul_ps(a0, b);
      __m256 acc21 = _mm256_mul_ps(a1, b);

        b = Bcol[3];
      __m256 acc30 = _mm256_mul_ps(a0, b);
      __m256 acc31 = _mm256_mul_ps(a1, b);

        b = Bcol[4];
      __m256 acc40 = _mm256_mul_ps(a0, b);
      __m256 acc41 = _mm256_mul_ps(a1, b);

      for (int k = 1; k < bb_nRows; k += 2) {
        Bcol += SIMD_ELEM_PEC_COL;
        a0 = _mm256_broadcast_ss(&A[k]);
        a1 = _mm256_broadcast_ss(&A[k+lda]);

        b = Bcol[0];
        acc00 = _mm256_fmadd_ps(a0, b, acc00);
        acc01 = _mm256_fmadd_ps(a1, b, acc01);

        b = Bcol[1];
        acc10 = _mm256_fmadd_ps(a0, b, acc10);
        acc11 = _mm256_fmadd_ps(a1, b, acc11);

        b = Bcol[2];
        acc20 = _mm256_fmadd_ps(a0, b, acc20);
        acc21 = _mm256_fmadd_ps(a1, b, acc21);

        b = Bcol[3];
        acc30 = _mm256_fmadd_ps(a0, b, acc30);
        acc31 = _mm256_fmadd_ps(a1, b, acc31);

        b = Bcol[4];
        acc40 = _mm256_fmadd_ps(a0, b, acc40);
        acc41 = _mm256_fmadd_ps(a1, b, acc41);

        Bcol += SIMD_ELEM_PEC_COL;
        a0 = _mm256_broadcast_ss(&A[k+1]);
        a1 = _mm256_broadcast_ss(&A[k+lda+1]);

        b = Bcol[0];
        acc00 = _mm256_fmadd_ps(a0, b, acc00);
        acc01 = _mm256_fmadd_ps(a1, b, acc01);

        b = Bcol[1];
        acc10 = _mm256_fmadd_ps(a0, b, acc10);
        acc11 = _mm256_fmadd_ps(a1, b, acc11);

        b = Bcol[2];
        acc20 = _mm256_fmadd_ps(a0, b, acc20);
        acc21 = _mm256_fmadd_ps(a1, b, acc21);

        b = Bcol[3];
        acc30 = _mm256_fmadd_ps(a0, b, acc30);
        acc31 = _mm256_fmadd_ps(a1, b, acc31);

        b = Bcol[4];
        acc40 = _mm256_fmadd_ps(a0, b, acc40);
        acc41 = _mm256_fmadd_ps(a1, b, acc41);
      }
      __m256 alpha_ps = _mm256_broadcast_ss(&pPrm->alpha);

      _mm256_storeu_ps(&Crow0[SIMD_FACTOR*0], _mm256_fmadd_ps(acc00, alpha_ps, _mm256_loadu_ps(&Crow0[SIMD_FACTOR*0])));
      _mm256_storeu_ps(&Crow0[SIMD_FACTOR*1], _mm256_fmadd_ps(acc10, alpha_ps, _mm256_loadu_ps(&Crow0[SIMD_FACTOR*1])));
      _mm256_storeu_ps(&Crow0[SIMD_FACTOR*2], _mm256_fmadd_ps(acc20, alpha_ps, _mm256_loadu_ps(&Crow0[SIMD_FACTOR*2])));
      _mm256_storeu_ps(&Crow0[SIMD_FACTOR*3], _mm256_fmadd_ps(acc30, alpha_ps, _mm256_loadu_ps(&Crow0[SIMD_FACTOR*3])));
      _mm256_storeu_ps(&Crow0[SIMD_FACTOR*4], _mm256_fmadd_ps(acc40, alpha_ps, _mm256_loadu_ps(&Crow0[SIMD_FACTOR*4])));

      float* Crow1 = Crow0+ldc;
      _mm256_storeu_ps(&Crow1[SIMD_FACTOR*0], _mm256_fmadd_ps(acc01, alpha_ps, _mm256_loadu_ps(&Crow1[SIMD_FACTOR*0])));
      _mm256_storeu_ps(&Crow1[SIMD_FACTOR*1], _mm256_fmadd_ps(acc11, alpha_ps, _mm256_loadu_ps(&Crow1[SIMD_FACTOR*1])));
      _mm256_storeu_ps(&Crow1[SIMD_FACTOR*2], _mm256_fmadd_ps(acc21, alpha_ps, _mm256_loadu_ps(&Crow1[SIMD_FACTOR*2])));
      _mm256_storeu_ps(&Crow1[SIMD_FACTOR*3], _mm256_fmadd_ps(acc31, alpha_ps, _mm256_loadu_ps(&Crow1[SIMD_FACTOR*3])));
      _mm256_storeu_ps(&Crow1[SIMD_FACTOR*4], _mm256_fmadd_ps(acc41, alpha_ps, _mm256_loadu_ps(&Crow1[SIMD_FACTOR*4])));

      Crow0 += COLS_PER_LOOP*SIMD_FACTOR;
    }
  }
  if (m < pPrm->M) {
    float* Crow0 = C;
    for (int n = 0; n < SIMD_ELEM_PEC_COL; n += COLS_PER_LOOP) {
      const __m256 *Bcol = &pPrm->bb[n];
      __m256 acc00 = _mm256_setzero_ps();
      __m256 acc10 = _mm256_setzero_ps();
      __m256 acc20 = _mm256_setzero_ps();
      __m256 acc30 = _mm256_setzero_ps();
      __m256 acc40 = _mm256_setzero_ps();
      for (int k = 0; k < bb_nRows; ++k) {
        __m256 a0 = _mm256_broadcast_ss(&A[k]);
        __m256 b;

        b = Bcol[0];
        acc00 = _mm256_add_ps(acc00, _mm256_mul_ps(a0, b));

        b = Bcol[1];
        acc10 = _mm256_add_ps(acc10, _mm256_mul_ps(a0, b));

        b = Bcol[2];
        acc20 = _mm256_add_ps(acc20, _mm256_mul_ps(a0, b));

        b = Bcol[3];
        acc30 = _mm256_add_ps(acc30, _mm256_mul_ps(a0, b));

        b = Bcol[4];
        acc40 = _mm256_add_ps(acc40, _mm256_mul_ps(a0, b));

        Bcol += SIMD_ELEM_PEC_COL;
      }
      __m256 alpha_ps = _mm256_broadcast_ss(&pPrm->alpha);

      _mm256_storeu_ps(&Crow0[SIMD_FACTOR*0], _mm256_fmadd_ps(acc00, alpha_ps, _mm256_loadu_ps(&Crow0[SIMD_FACTOR*0])));
      _mm256_storeu_ps(&Crow0[SIMD_FACTOR*1], _mm256_fmadd_ps(acc10, alpha_ps, _mm256_loadu_ps(&Crow0[SIMD_FACTOR*1])));
      _mm256_storeu_ps(&Crow0[SIMD_FACTOR*2], _mm256_fmadd_ps(acc20, alpha_ps, _mm256_loadu_ps(&Crow0[SIMD_FACTOR*2])));
      _mm256_storeu_ps(&Crow0[SIMD_FACTOR*3], _mm256_fmadd_ps(acc30, alpha_ps, _mm256_loadu_ps(&Crow0[SIMD_FACTOR*3])));
      _mm256_storeu_ps(&Crow0[SIMD_FACTOR*4], _mm256_fmadd_ps(acc40, alpha_ps, _mm256_loadu_ps(&Crow0[SIMD_FACTOR*4])));

      Crow0 += COLS_PER_LOOP*SIMD_FACTOR;
    }
  }
}

static void fma256_noncblas_sgemm_core_bottomRows(
 const noncblas_sgemm_prm_t* pPrm,
 const float *A,
 float *C,
 int nRows)
{
  int lda = pPrm->lda;
  int ldc = pPrm->ldc;
  int m;
  for (m = 0; m < pPrm->M-1; A += lda*A_WORDS_PER_ITER, C += ldc*A_WORDS_PER_ITER, m += A_WORDS_PER_ITER) {
    float* Crow0 = C;
    for (int n = 0; n < SIMD_ELEM_PEC_COL; n += COLS_PER_LOOP) {
      __m256 a0 = _mm256_broadcast_ss(&A[0]);
      __m256 a1 = _mm256_broadcast_ss(&A[lda]);
      const __m256 *Bcol = &pPrm->bb[n];
      __m256 b;
        b = Bcol[0];
      __m256 acc00 = _mm256_mul_ps(a0, b);
      __m256 acc01 = _mm256_mul_ps(a1, b);

        b = Bcol[1];
      __m256 acc10 = _mm256_mul_ps(a0, b);
      __m256 acc11 = _mm256_mul_ps(a1, b);

        b = Bcol[2];
      __m256 acc20 = _mm256_mul_ps(a0, b);
      __m256 acc21 = _mm256_mul_ps(a1, b);

        b = Bcol[3];
      __m256 acc30 = _mm256_mul_ps(a0, b);
      __m256 acc31 = _mm256_mul_ps(a1, b);

        b = Bcol[4];
      __m256 acc40 = _mm256_mul_ps(a0, b);
      __m256 acc41 = _mm256_mul_ps(a1, b);

      Bcol += SIMD_ELEM_PEC_COL;
      for (int k = 1; k < nRows; ++k) {
        a0 = _mm256_broadcast_ss(&A[k]);
        a1 = _mm256_broadcast_ss(&A[k+lda]);

        b = Bcol[0];
        acc00 = _mm256_fmadd_ps(a0, b, acc00);
        acc01 = _mm256_fmadd_ps(a1, b, acc01);

        b = Bcol[1];
        acc10 = _mm256_fmadd_ps(a0, b, acc10);
        acc11 = _mm256_fmadd_ps(a1, b, acc11);

        b = Bcol[2];
        acc20 = _mm256_fmadd_ps(a0, b, acc20);
        acc21 = _mm256_fmadd_ps(a1, b, acc21);

        b = Bcol[3];
        acc30 = _mm256_fmadd_ps(a0, b, acc30);
        acc31 = _mm256_fmadd_ps(a1, b, acc31);

        b = Bcol[4];
        acc40 = _mm256_fmadd_ps(a0, b, acc40);
        acc41 = _mm256_fmadd_ps(a1, b, acc41);

        Bcol += SIMD_ELEM_PEC_COL;
      }
      __m256 alpha_ps = _mm256_broadcast_ss(&pPrm->alpha);

      _mm256_storeu_ps(&Crow0[SIMD_FACTOR*0], _mm256_fmadd_ps(acc00, alpha_ps, _mm256_loadu_ps(&Crow0[SIMD_FACTOR*0])));
      _mm256_storeu_ps(&Crow0[SIMD_FACTOR*1], _mm256_fmadd_ps(acc10, alpha_ps, _mm256_loadu_ps(&Crow0[SIMD_FACTOR*1])));
      _mm256_storeu_ps(&Crow0[SIMD_FACTOR*2], _mm256_fmadd_ps(acc20, alpha_ps, _mm256_loadu_ps(&Crow0[SIMD_FACTOR*2])));
      _mm256_storeu_ps(&Crow0[SIMD_FACTOR*3], _mm256_fmadd_ps(acc30, alpha_ps, _mm256_loadu_ps(&Crow0[SIMD_FACTOR*3])));
      _mm256_storeu_ps(&Crow0[SIMD_FACTOR*4], _mm256_fmadd_ps(acc40, alpha_ps, _mm256_loadu_ps(&Crow0[SIMD_FACTOR*4])));

      float* Crow1 = Crow0+ldc;
      _mm256_storeu_ps(&Crow1[SIMD_FACTOR*0], _mm256_fmadd_ps(acc01, alpha_ps, _mm256_loadu_ps(&Crow1[SIMD_FACTOR*0])));
      _mm256_storeu_ps(&Crow1[SIMD_FACTOR*1], _mm256_fmadd_ps(acc11, alpha_ps, _mm256_loadu_ps(&Crow1[SIMD_FACTOR*1])));
      _mm256_storeu_ps(&Crow1[SIMD_FACTOR*2], _mm256_fmadd_ps(acc21, alpha_ps, _mm256_loadu_ps(&Crow1[SIMD_FACTOR*2])));
      _mm256_storeu_ps(&Crow1[SIMD_FACTOR*3], _mm256_fmadd_ps(acc31, alpha_ps, _mm256_loadu_ps(&Crow1[SIMD_FACTOR*3])));
      _mm256_storeu_ps(&Crow1[SIMD_FACTOR*4], _mm256_fmadd_ps(acc41, alpha_ps, _mm256_loadu_ps(&Crow1[SIMD_FACTOR*4])));

      Crow0 += COLS_PER_LOOP*SIMD_FACTOR;
    }
  }
  if (m < pPrm->M) {
    float* Crow0 = C;
    for (int n = 0; n < SIMD_ELEM_PEC_COL; n += COLS_PER_LOOP) {
      const __m256 *Bcol = &pPrm->bb[n];
      __m256 acc00 = _mm256_setzero_ps();
      __m256 acc10 = _mm256_setzero_ps();
      __m256 acc20 = _mm256_setzero_ps();
      __m256 acc30 = _mm256_setzero_ps();
      __m256 acc40 = _mm256_setzero_ps();
      for (int k = 0; k < nRows; ++k) {
        __m256 a0 = _mm256_broadcast_ss(&A[k]);
        __m256 b;

        b = Bcol[0];
        acc00 = _mm256_add_ps(acc00, _mm256_mul_ps(a0, b));

        b = Bcol[1];
        acc10 = _mm256_add_ps(acc10, _mm256_mul_ps(a0, b));

        b = Bcol[2];
        acc20 = _mm256_add_ps(acc20, _mm256_mul_ps(a0, b));

        b = Bcol[3];
        acc30 = _mm256_add_ps(acc30, _mm256_mul_ps(a0, b));

        b = Bcol[4];
        acc40 = _mm256_add_ps(acc40, _mm256_mul_ps(a0, b));

        Bcol += SIMD_ELEM_PEC_COL;
      }
      __m256 alpha_ps = _mm256_broadcast_ss(&pPrm->alpha);

      _mm256_storeu_ps(&Crow0[SIMD_FACTOR*0], _mm256_fmadd_ps(acc00, alpha_ps, _mm256_loadu_ps(&Crow0[SIMD_FACTOR*0])));
      _mm256_storeu_ps(&Crow0[SIMD_FACTOR*1], _mm256_fmadd_ps(acc10, alpha_ps, _mm256_loadu_ps(&Crow0[SIMD_FACTOR*1])));
      _mm256_storeu_ps(&Crow0[SIMD_FACTOR*2], _mm256_fmadd_ps(acc20, alpha_ps, _mm256_loadu_ps(&Crow0[SIMD_FACTOR*2])));
      _mm256_storeu_ps(&Crow0[SIMD_FACTOR*3], _mm256_fmadd_ps(acc30, alpha_ps, _mm256_loadu_ps(&Crow0[SIMD_FACTOR*3])));
      _mm256_storeu_ps(&Crow0[SIMD_FACTOR*4], _mm256_fmadd_ps(acc40, alpha_ps, _mm256_loadu_ps(&Crow0[SIMD_FACTOR*4])));

      Crow0 += COLS_PER_LOOP*SIMD_FACTOR;
    }
  }
}

static int fma256_noncblas_sgemm_core_rightmostColumns_mainLoop(
 const noncblas_sgemm_prm_t* pPrm,
 const float *A,
 float *C,
 int nCols, // 0 < nCols <  bb_nCols
 int nRows)
{
  int nn  = (nCols / (COLS_PER_LOOP*SIMD_FACTOR))*COLS_PER_LOOP;
  int lda = pPrm->lda;
  int ldc = pPrm->ldc;
  int m;
  for (m = 0; m < pPrm->M-1; A += lda*A_WORDS_PER_ITER, C += ldc*A_WORDS_PER_ITER, m += A_WORDS_PER_ITER) {
    float* Crow0 = C;
    for (int n = 0; n < nn; n += COLS_PER_LOOP) {
      __m256 a0 = _mm256_broadcast_ss(&A[0]);
      __m256 a1 = _mm256_broadcast_ss(&A[lda]);
      const __m256 *Bcol = &pPrm->bb[n];
      __m256 b;
        b = Bcol[0];
      __m256 acc00 = _mm256_mul_ps(a0, b);
      __m256 acc01 = _mm256_mul_ps(a1, b);

        b = Bcol[1];
      __m256 acc10 = _mm256_mul_ps(a0, b);
      __m256 acc11 = _mm256_mul_ps(a1, b);

        b = Bcol[2];
      __m256 acc20 = _mm256_mul_ps(a0, b);
      __m256 acc21 = _mm256_mul_ps(a1, b);

        b = Bcol[3];
      __m256 acc30 = _mm256_mul_ps(a0, b);
      __m256 acc31 = _mm256_mul_ps(a1, b);

        b = Bcol[4];
      __m256 acc40 = _mm256_mul_ps(a0, b);
      __m256 acc41 = _mm256_mul_ps(a1, b);

      Bcol += SIMD_ELEM_PEC_COL;
      for (int k = 1; k < nRows; ++k) {
        a0 = _mm256_broadcast_ss(&A[k]);
        a1 = _mm256_broadcast_ss(&A[k+lda]);

        b = Bcol[0];
        acc00 = _mm256_fmadd_ps(a0, b, acc00);
        acc01 = _mm256_fmadd_ps(a1, b, acc01);

        b = Bcol[1];
        acc10 = _mm256_fmadd_ps(a0, b, acc10);
        acc11 = _mm256_fmadd_ps(a1, b, acc11);

        b = Bcol[2];
        acc20 = _mm256_fmadd_ps(a0, b, acc20);
        acc21 = _mm256_fmadd_ps(a1, b, acc21);

        b = Bcol[3];
        acc30 = _mm256_fmadd_ps(a0, b, acc30);
        acc31 = _mm256_fmadd_ps(a1, b, acc31);

        b = Bcol[4];
        acc40 = _mm256_fmadd_ps(a0, b, acc40);
        acc41 = _mm256_fmadd_ps(a1, b, acc41);

        Bcol += SIMD_ELEM_PEC_COL;
      }
      __m256 alpha_ps = _mm256_broadcast_ss(&pPrm->alpha);

      _mm256_storeu_ps(&Crow0[SIMD_FACTOR*0], _mm256_fmadd_ps(acc00, alpha_ps, _mm256_loadu_ps(&Crow0[SIMD_FACTOR*0])));
      _mm256_storeu_ps(&Crow0[SIMD_FACTOR*1], _mm256_fmadd_ps(acc10, alpha_ps, _mm256_loadu_ps(&Crow0[SIMD_FACTOR*1])));
      _mm256_storeu_ps(&Crow0[SIMD_FACTOR*2], _mm256_fmadd_ps(acc20, alpha_ps, _mm256_loadu_ps(&Crow0[SIMD_FACTOR*2])));
      _mm256_storeu_ps(&Crow0[SIMD_FACTOR*3], _mm256_fmadd_ps(acc30, alpha_ps, _mm256_loadu_ps(&Crow0[SIMD_FACTOR*3])));
      _mm256_storeu_ps(&Crow0[SIMD_FACTOR*4], _mm256_fmadd_ps(acc40, alpha_ps, _mm256_loadu_ps(&Crow0[SIMD_FACTOR*4])));

      float* Crow1 = Crow0+ldc;
      _mm256_storeu_ps(&Crow1[SIMD_FACTOR*0], _mm256_fmadd_ps(acc01, alpha_ps, _mm256_loadu_ps(&Crow1[SIMD_FACTOR*0])));
      _mm256_storeu_ps(&Crow1[SIMD_FACTOR*1], _mm256_fmadd_ps(acc11, alpha_ps, _mm256_loadu_ps(&Crow1[SIMD_FACTOR*1])));
      _mm256_storeu_ps(&Crow1[SIMD_FACTOR*2], _mm256_fmadd_ps(acc21, alpha_ps, _mm256_loadu_ps(&Crow1[SIMD_FACTOR*2])));
      _mm256_storeu_ps(&Crow1[SIMD_FACTOR*3], _mm256_fmadd_ps(acc31, alpha_ps, _mm256_loadu_ps(&Crow1[SIMD_FACTOR*3])));
      _mm256_storeu_ps(&Crow1[SIMD_FACTOR*4], _mm256_fmadd_ps(acc41, alpha_ps, _mm256_loadu_ps(&Crow1[SIMD_FACTOR*4])));

      Crow0 += COLS_PER_LOOP*SIMD_FACTOR;
    }
  }
  if (m < pPrm->M) {
    float* Crow0 = C;
    for (int n = 0; n < nn; n += COLS_PER_LOOP) {
      const __m256 *Bcol = &pPrm->bb[n];
      __m256 acc00 = _mm256_setzero_ps();
      __m256 acc10 = _mm256_setzero_ps();
      __m256 acc20 = _mm256_setzero_ps();
      __m256 acc30 = _mm256_setzero_ps();
      __m256 acc40 = _mm256_setzero_ps();
      for (int k = 0; k < nRows; ++k) {
        __m256 a0 = _mm256_broadcast_ss(&A[k]);
        __m256 b;

        b = Bcol[0];
        acc00 = _mm256_add_ps(acc00, _mm256_mul_ps(a0, b));

        b = Bcol[1];
        acc10 = _mm256_add_ps(acc10, _mm256_mul_ps(a0, b));

        b = Bcol[2];
        acc20 = _mm256_add_ps(acc20, _mm256_mul_ps(a0, b));

        b = Bcol[3];
        acc30 = _mm256_add_ps(acc30, _mm256_mul_ps(a0, b));

        b = Bcol[4];
        acc40 = _mm256_add_ps(acc40, _mm256_mul_ps(a0, b));

        Bcol += SIMD_ELEM_PEC_COL;
      }
      __m256 alpha_ps = _mm256_broadcast_ss(&pPrm->alpha);

      _mm256_storeu_ps(&Crow0[SIMD_FACTOR*0], _mm256_fmadd_ps(acc00, alpha_ps, _mm256_loadu_ps(&Crow0[SIMD_FACTOR*0])));
      _mm256_storeu_ps(&Crow0[SIMD_FACTOR*1], _mm256_fmadd_ps(acc10, alpha_ps, _mm256_loadu_ps(&Crow0[SIMD_FACTOR*1])));
      _mm256_storeu_ps(&Crow0[SIMD_FACTOR*2], _mm256_fmadd_ps(acc20, alpha_ps, _mm256_loadu_ps(&Crow0[SIMD_FACTOR*2])));
      _mm256_storeu_ps(&Crow0[SIMD_FACTOR*3], _mm256_fmadd_ps(acc30, alpha_ps, _mm256_loadu_ps(&Crow0[SIMD_FACTOR*3])));
      _mm256_storeu_ps(&Crow0[SIMD_FACTOR*4], _mm256_fmadd_ps(acc40, alpha_ps, _mm256_loadu_ps(&Crow0[SIMD_FACTOR*4])));

      Crow0 += COLS_PER_LOOP*SIMD_FACTOR;
    }
  }
  return nn;
}


static void fma256_noncblas_sgemm_core_rightmostColumns(
 noncblas_sgemm_prm_t* pPrm,
 const float*          A,
 float*                C,
 int                   nCols, // 0 < nCols <  bb_nCols
 int                   nRows) // nRows <= bb_nRows
{
  int simdWordsProcessed = 0;
  if (nCols >= COLS_PER_LOOP*SIMD_FACTOR)
    simdWordsProcessed = fma256_noncblas_sgemm_core_rightmostColumns_mainLoop(pPrm, A, C, nCols, nRows);
  nCols -= simdWordsProcessed*SIMD_FACTOR;
  C     += simdWordsProcessed*SIMD_FACTOR;

  int lda = pPrm->lda;
  int ldc = pPrm->ldc;
  int m;
  for (m = 0; m < pPrm->M - A_WORDS_PER_ITER + 1;
    C += ldc*A_WORDS_PER_ITER,
    A += lda*A_WORDS_PER_ITER,
    m += A_WORDS_PER_ITER) {
    memcpy(&pPrm->cc[0 * COLS_PER_LOOP], &C[ldc * 0], nCols*sizeof(*C));
    memcpy(&pPrm->cc[1 * COLS_PER_LOOP], &C[ldc * 1], nCols*sizeof(*C));

    const __m256 *Bcol = &pPrm->bb[simdWordsProcessed];
    __m256 a0 = _mm256_broadcast_ss(&A[lda * 0 + 0]);
    __m256 a1 = _mm256_broadcast_ss(&A[lda * 1 + 0]);
    __m256 b;

    b = Bcol[0];
    __m256 acc00 = _mm256_mul_ps(a0, b);
    __m256 acc01 = _mm256_mul_ps(a1, b);

    b = Bcol[1];
    __m256 acc10 = _mm256_mul_ps(a0, b);
    __m256 acc11 = _mm256_mul_ps(a1, b);

    b = Bcol[2];
    __m256 acc20 = _mm256_mul_ps(a0, b);
    __m256 acc21 = _mm256_mul_ps(a1, b);

    b = Bcol[3];
    __m256 acc30 = _mm256_mul_ps(a0, b);
    __m256 acc31 = _mm256_mul_ps(a1, b);

    b = Bcol[4];
    __m256 acc40 = _mm256_mul_ps(a0, b);
    __m256 acc41 = _mm256_mul_ps(a1, b);

    Bcol += SIMD_ELEM_PEC_COL;

    for (int k = 1; k < nRows; ++k) {
      a0 = _mm256_broadcast_ss(&A[lda * 0 + k]);
      a1 = _mm256_broadcast_ss(&A[lda * 1 + k]);

      b = Bcol[0];
      acc00 = _mm256_fmadd_ps(a0, b, acc00);
      acc01 = _mm256_fmadd_ps(a1, b, acc01);


      b = Bcol[1];
      acc10 = _mm256_fmadd_ps(a0, b, acc10);
      acc11 = _mm256_fmadd_ps(a1, b, acc11);

      b = Bcol[2];
      acc20 = _mm256_fmadd_ps(a0, b, acc20);
      acc21 = _mm256_fmadd_ps(a1, b, acc21);

      b = Bcol[3];
      acc30 = _mm256_fmadd_ps(a0, b, acc30);
      acc31 = _mm256_fmadd_ps(a1, b, acc31);

      b = Bcol[4];
      acc40 = _mm256_fmadd_ps(a0, b, acc40);
      acc41 = _mm256_fmadd_ps(a1, b, acc41);

      Bcol += SIMD_ELEM_PEC_COL;
    }

    __m256 alpha_ps = _mm256_broadcast_ss(&pPrm->alpha);
    pPrm->cc[0 * COLS_PER_LOOP + 0] = _mm256_fmadd_ps(acc00, alpha_ps, pPrm->cc[0 * COLS_PER_LOOP + 0]);
    pPrm->cc[0 * COLS_PER_LOOP + 1] = _mm256_fmadd_ps(acc10, alpha_ps, pPrm->cc[0 * COLS_PER_LOOP + 1]);
    pPrm->cc[0 * COLS_PER_LOOP + 2] = _mm256_fmadd_ps(acc20, alpha_ps, pPrm->cc[0 * COLS_PER_LOOP + 2]);
    pPrm->cc[0 * COLS_PER_LOOP + 3] = _mm256_fmadd_ps(acc30, alpha_ps, pPrm->cc[0 * COLS_PER_LOOP + 3]);
    pPrm->cc[0 * COLS_PER_LOOP + 4] = _mm256_fmadd_ps(acc40, alpha_ps, pPrm->cc[0 * COLS_PER_LOOP + 4]);

    pPrm->cc[1 * COLS_PER_LOOP + 0] = _mm256_fmadd_ps(acc01, alpha_ps, pPrm->cc[1 * COLS_PER_LOOP + 0]);
    pPrm->cc[1 * COLS_PER_LOOP + 1] = _mm256_fmadd_ps(acc11, alpha_ps, pPrm->cc[1 * COLS_PER_LOOP + 1]);
    pPrm->cc[1 * COLS_PER_LOOP + 2] = _mm256_fmadd_ps(acc21, alpha_ps, pPrm->cc[1 * COLS_PER_LOOP + 2]);
    pPrm->cc[1 * COLS_PER_LOOP + 3] = _mm256_fmadd_ps(acc31, alpha_ps, pPrm->cc[1 * COLS_PER_LOOP + 3]);
    pPrm->cc[1 * COLS_PER_LOOP + 4] = _mm256_fmadd_ps(acc41, alpha_ps, pPrm->cc[1 * COLS_PER_LOOP + 4]);

    memcpy(&C[ldc * 0], &pPrm->cc[0 * COLS_PER_LOOP], nCols*sizeof(*C));
    memcpy(&C[ldc * 1], &pPrm->cc[1 * COLS_PER_LOOP], nCols*sizeof(*C));
  }

  if (m != pPrm->M) {
    // process a bottom row of A
    memcpy(&pPrm->cc[0], C, nCols*sizeof(*C));
    __m256 a = _mm256_broadcast_ss(&A[0]);
    const __m256 *Bcol = &pPrm->bb[simdWordsProcessed];
    __m256 acc00 = _mm256_mul_ps(a, Bcol[0]);
    __m256 acc10 = _mm256_mul_ps(a, Bcol[1]);
    __m256 acc20 = _mm256_mul_ps(a, Bcol[2]);
    __m256 acc30 = _mm256_mul_ps(a, Bcol[3]);
    __m256 acc40 = _mm256_mul_ps(a, Bcol[4]);
    Bcol += SIMD_ELEM_PEC_COL;

    // It is easy to better here, but it is almost certainly does not matter
    for (int k = 1; k < nRows; ++k) {
      a = _mm256_broadcast_ss(&A[k]);
      acc00 = _mm256_fmadd_ps(a, Bcol[0], acc00);
      acc10 = _mm256_fmadd_ps(a, Bcol[1], acc10);
      acc20 = _mm256_fmadd_ps(a, Bcol[2], acc20);
      acc30 = _mm256_fmadd_ps(a, Bcol[3], acc30);
      acc40 = _mm256_fmadd_ps(a, Bcol[4], acc40);
      Bcol += SIMD_ELEM_PEC_COL;
    }

    __m256 alpha_ps = _mm256_broadcast_ss(&pPrm->alpha);
    pPrm->cc[0] = _mm256_fmadd_ps(acc00, alpha_ps, pPrm->cc[0]);
    pPrm->cc[1] = _mm256_fmadd_ps(acc10, alpha_ps, pPrm->cc[1]);
    pPrm->cc[2] = _mm256_fmadd_ps(acc20, alpha_ps, pPrm->cc[2]);
    pPrm->cc[3] = _mm256_fmadd_ps(acc30, alpha_ps, pPrm->cc[3]);
    pPrm->cc[4] = _mm256_fmadd_ps(acc30, alpha_ps, pPrm->cc[4]);
    memcpy(C, pPrm->cc, nCols*sizeof(*C));
  }
}


static void fma256_noncblas_sgemm_multC(
 int M, int N,
 float beta,
 float *C, int ldc)
{
  if (beta != 0) {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n)
        C[n] *= beta;
      C += ldc;
    }
  } else {
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < N; ++n)
        C[n] = 0;
      C += ldc;
    }
  }
}

void fma256_noncblas_sgemm(
  int M, int N, int K,
  float alpha,
  const float *A, int lda,
  const float *B, int ldb,
  float beta,
  float *C, int ldc)
{
  noncblas_sgemm_prm_t prm;
  prm.lda = lda;
  prm.ldc = ldc;
  prm.alpha = alpha;
  memset(prm.cc, 0, sizeof(prm.cc));

  const int m_step_nom = 200;
  const int m_step_max = 320;
  int n_Rsteps = K / bb_nRows;
  int n_Csteps = N / bb_nCols;
  if (n_Csteps == 0)
    memset(prm.bb, 0, sizeof(prm.bb));
  for (int m = 0; m < M; m += prm.M) {
    prm.M = M - m <= m_step_max ? M - m : m_step_nom;

    float *Crow = &C[m*ldc];
    fma256_noncblas_sgemm_multC(prm.M, N, beta, Crow, ldc);

    const float *Arow = &A[m*lda];
    int row = 0;
    for (int ri = 0; ri < n_Rsteps; ++ri) {
      int col = 0;
      for (int ci = 0; ci < n_Csteps; ++ci) {
        // process full rectangles
        const float* bSrc = &B[row*ldb + col];
        for (int i = 0; i < bb_nRows; ++i) {
          memcpy(&prm.bb[SIMD_ELEM_PEC_COL*i], bSrc, bb_nCols*sizeof(*B));
          bSrc += ldb;
        }
        fma256_noncblas_sgemm_core(&prm, &Arow[row], &Crow[col]);
        col += bb_nCols;
      }
      if (col < N) {
        // process rightmost rectangle of the full-height band
        const float* bSrc = &B[row*ldb + col];
        for (int i = 0; i < bb_nRows; ++i) {
          memcpy(&prm.bb[SIMD_ELEM_PEC_COL*i], bSrc, (N - col)*sizeof(*B));
          bSrc += ldb;
        }
        fma256_noncblas_sgemm_core_rightmostColumns(&prm, &Arow[row], &Crow[col], N-col, bb_nRows);
      }
      row += bb_nRows;
    }
    if (row < K) {
      // bottom band
      int col = 0;
      for (int ci = 0; ci < n_Csteps; ++ci) {
        // process full-width rectangles
        const float* bSrc = &B[row*ldb + col];
        for (int i = 0; i < K-row; ++i) {
          memcpy(&prm.bb[SIMD_ELEM_PEC_COL*i], bSrc, bb_nCols*sizeof(*B));
          bSrc += ldb;
        }
        fma256_noncblas_sgemm_core_bottomRows(&prm, &Arow[row], &Crow[col], K-row);
        col += bb_nCols;
      }
      if (col < N) {
        // process bottom-right corner rectangle
        const float* bSrc = &B[row*ldb + col];
        for (int i = 0; i < K-row; ++i) {
          memcpy(&prm.bb[SIMD_ELEM_PEC_COL*i], bSrc, (N-col)*sizeof(*B));
          bSrc += ldb;
        }
        fma256_noncblas_sgemm_core_rightmostColumns(&prm, &Arow[row], &Crow[col], N-col, K-row);
      }
    }
  }
}
