#include <string.h>
#include <intrin.h>

enum {
 SIMD_FACTOR           = 8,
 B_WORDS_PER_ITER      = 4,
 A_WORDS_PER_ITER      = 2,
 bb_nCols              = B_WORDS_PER_ITER*SIMD_FACTOR,
 bb_nRows              = 128,
};

struct noncblas_sgemm_prm_t {
  int    M;
  int    lda;
  int    ldc;
  float  alpha;
  __m256 bb[B_WORDS_PER_ITER*bb_nRows];
  __m256 cc[B_WORDS_PER_ITER*A_WORDS_PER_ITER];
};

static void fma256_noncblas_sgemm_core(noncblas_sgemm_prm_t* pPrm, const float  *A, float *C)
{
  int lda = pPrm->lda;
  int ldc = pPrm->ldc;
  int m;
  for (m = 0; m < pPrm->M - A_WORDS_PER_ITER + 1;
    C += ldc*A_WORDS_PER_ITER,
    A += lda*A_WORDS_PER_ITER,
    m += A_WORDS_PER_ITER) {
    const __m256 *Bcol = pPrm->bb;
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

    Bcol += B_WORDS_PER_ITER;

    for (int k = 1; k < bb_nRows; ++k) {
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

      Bcol += B_WORDS_PER_ITER;
    }

    __m256 alpha_ps = _mm256_broadcast_ss(&pPrm->alpha);
    float* Ccol = C;

    _mm256_storeu_ps(&Ccol[SIMD_FACTOR * 0], _mm256_fmadd_ps(acc00, alpha_ps, _mm256_loadu_ps(&Ccol[SIMD_FACTOR * 0])));
    _mm256_storeu_ps(&Ccol[SIMD_FACTOR * 1], _mm256_fmadd_ps(acc10, alpha_ps, _mm256_loadu_ps(&Ccol[SIMD_FACTOR * 1])));
    _mm256_storeu_ps(&Ccol[SIMD_FACTOR * 2], _mm256_fmadd_ps(acc20, alpha_ps, _mm256_loadu_ps(&Ccol[SIMD_FACTOR * 2])));
    _mm256_storeu_ps(&Ccol[SIMD_FACTOR * 3], _mm256_fmadd_ps(acc30, alpha_ps, _mm256_loadu_ps(&Ccol[SIMD_FACTOR * 3])));
    Ccol += ldc;

    _mm256_storeu_ps(&Ccol[SIMD_FACTOR * 0], _mm256_fmadd_ps(acc01, alpha_ps, _mm256_loadu_ps(&Ccol[SIMD_FACTOR * 0])));
    _mm256_storeu_ps(&Ccol[SIMD_FACTOR * 1], _mm256_fmadd_ps(acc11, alpha_ps, _mm256_loadu_ps(&Ccol[SIMD_FACTOR * 1])));
    _mm256_storeu_ps(&Ccol[SIMD_FACTOR * 2], _mm256_fmadd_ps(acc21, alpha_ps, _mm256_loadu_ps(&Ccol[SIMD_FACTOR * 2])));
    _mm256_storeu_ps(&Ccol[SIMD_FACTOR * 3], _mm256_fmadd_ps(acc31, alpha_ps, _mm256_loadu_ps(&Ccol[SIMD_FACTOR * 3])));
  }

  if (m != pPrm->M) {
    // process a bottom row of A
    const __m256 *Bcol = pPrm->bb;
    __m256 b0 = Bcol[0];
    __m256 b1 = Bcol[1];
    __m256 b2 = Bcol[2];
    __m256 b3 = Bcol[3];
    Bcol += B_WORDS_PER_ITER;
    __m256 a;

    a = _mm256_broadcast_ss(&A[0]);
    __m256 acc00 = _mm256_mul_ps(a, b0);
    __m256 acc10 = _mm256_mul_ps(a, b1);
    __m256 acc20 = _mm256_mul_ps(a, b2);
    __m256 acc30 = _mm256_mul_ps(a, b3);

    // It is easy to better here, but let's leave it for later
    for (int k = 1; k < bb_nRows; ++k) {
      b0 = Bcol[0];
      b1 = Bcol[1];
      b2 = Bcol[2];
      b3 = Bcol[3];
      Bcol += B_WORDS_PER_ITER;

      a = _mm256_broadcast_ss(&A[k]);
      acc00 = _mm256_fmadd_ps(a, b0, acc00);
      acc10 = _mm256_fmadd_ps(a, b1, acc10);
      acc20 = _mm256_fmadd_ps(a, b2, acc20);
      acc30 = _mm256_fmadd_ps(a, b3, acc30);
    }

    __m256 alpha_ps = _mm256_broadcast_ss(&pPrm->alpha);
    float* Ccol = C;

    _mm256_storeu_ps(&Ccol[SIMD_FACTOR * 0], _mm256_fmadd_ps(acc00, alpha_ps, _mm256_loadu_ps(&Ccol[SIMD_FACTOR * 0])));
    _mm256_storeu_ps(&Ccol[SIMD_FACTOR * 1], _mm256_fmadd_ps(acc10, alpha_ps, _mm256_loadu_ps(&Ccol[SIMD_FACTOR * 1])));
    _mm256_storeu_ps(&Ccol[SIMD_FACTOR * 2], _mm256_fmadd_ps(acc20, alpha_ps, _mm256_loadu_ps(&Ccol[SIMD_FACTOR * 2])));
    _mm256_storeu_ps(&Ccol[SIMD_FACTOR * 3], _mm256_fmadd_ps(acc30, alpha_ps, _mm256_loadu_ps(&Ccol[SIMD_FACTOR * 3])));
  }
}

static void fma256_noncblas_sgemm_core_bottomRows(noncblas_sgemm_prm_t* pPrm, const float *A, float *C, int nRows)
{
  int lda = pPrm->lda;
  int ldc = pPrm->ldc;
  int m;
  for (m = 0; m < pPrm->M - A_WORDS_PER_ITER + 1;
    C += ldc*A_WORDS_PER_ITER,
    A += lda*A_WORDS_PER_ITER,
    m += A_WORDS_PER_ITER) {
    const __m256 *Bcol = pPrm->bb;
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

    Bcol += B_WORDS_PER_ITER;

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

      Bcol += B_WORDS_PER_ITER;
    }

    __m256 alpha_ps = _mm256_broadcast_ss(&pPrm->alpha);
    float* Ccol = C;

    _mm256_storeu_ps(&Ccol[SIMD_FACTOR * 0], _mm256_fmadd_ps(acc00, alpha_ps, _mm256_loadu_ps(&Ccol[SIMD_FACTOR * 0])));
    _mm256_storeu_ps(&Ccol[SIMD_FACTOR * 1], _mm256_fmadd_ps(acc10, alpha_ps, _mm256_loadu_ps(&Ccol[SIMD_FACTOR * 1])));
    _mm256_storeu_ps(&Ccol[SIMD_FACTOR * 2], _mm256_fmadd_ps(acc20, alpha_ps, _mm256_loadu_ps(&Ccol[SIMD_FACTOR * 2])));
    _mm256_storeu_ps(&Ccol[SIMD_FACTOR * 3], _mm256_fmadd_ps(acc30, alpha_ps, _mm256_loadu_ps(&Ccol[SIMD_FACTOR * 3])));
    Ccol += ldc;

    _mm256_storeu_ps(&Ccol[SIMD_FACTOR * 0], _mm256_fmadd_ps(acc01, alpha_ps, _mm256_loadu_ps(&Ccol[SIMD_FACTOR * 0])));
    _mm256_storeu_ps(&Ccol[SIMD_FACTOR * 1], _mm256_fmadd_ps(acc11, alpha_ps, _mm256_loadu_ps(&Ccol[SIMD_FACTOR * 1])));
    _mm256_storeu_ps(&Ccol[SIMD_FACTOR * 2], _mm256_fmadd_ps(acc21, alpha_ps, _mm256_loadu_ps(&Ccol[SIMD_FACTOR * 2])));
    _mm256_storeu_ps(&Ccol[SIMD_FACTOR * 3], _mm256_fmadd_ps(acc31, alpha_ps, _mm256_loadu_ps(&Ccol[SIMD_FACTOR * 3])));
  }

  if (m != pPrm->M) {
    // process a bottom row of A
    const __m256 *Bcol = pPrm->bb;
    __m256 b0 = Bcol[0];
    __m256 b1 = Bcol[1];
    __m256 b2 = Bcol[2];
    __m256 b3 = Bcol[3];
    Bcol += B_WORDS_PER_ITER;
    __m256 a;

    a = _mm256_broadcast_ss(&A[0]);
    __m256 acc00 = _mm256_mul_ps(a, b0);
    __m256 acc10 = _mm256_mul_ps(a, b1);
    __m256 acc20 = _mm256_mul_ps(a, b2);
    __m256 acc30 = _mm256_mul_ps(a, b3);

    // It is easy to better here, but let's leave it for later
    for (int k = 1; k < nRows; ++k) {
      b0 = Bcol[0];
      b1 = Bcol[1];
      b2 = Bcol[2];
      b3 = Bcol[3];
      Bcol += B_WORDS_PER_ITER;

      a = _mm256_broadcast_ss(&A[k]);
      acc00 = _mm256_fmadd_ps(a, b0, acc00);
      acc10 = _mm256_fmadd_ps(a, b1, acc10);
      acc20 = _mm256_fmadd_ps(a, b2, acc20);
      acc30 = _mm256_fmadd_ps(a, b3, acc30);
    }

    __m256 alpha_ps = _mm256_broadcast_ss(&pPrm->alpha);
    float* Ccol = C;

    _mm256_storeu_ps(&Ccol[SIMD_FACTOR * 0], _mm256_fmadd_ps(acc00, alpha_ps, _mm256_loadu_ps(&Ccol[SIMD_FACTOR * 0])));
    _mm256_storeu_ps(&Ccol[SIMD_FACTOR * 1], _mm256_fmadd_ps(acc10, alpha_ps, _mm256_loadu_ps(&Ccol[SIMD_FACTOR * 1])));
    _mm256_storeu_ps(&Ccol[SIMD_FACTOR * 2], _mm256_fmadd_ps(acc20, alpha_ps, _mm256_loadu_ps(&Ccol[SIMD_FACTOR * 2])));
    _mm256_storeu_ps(&Ccol[SIMD_FACTOR * 3], _mm256_fmadd_ps(acc30, alpha_ps, _mm256_loadu_ps(&Ccol[SIMD_FACTOR * 3])));
  }
}

static void fma256_noncblas_sgemm_core_rightmostColumns(
 noncblas_sgemm_prm_t* pPrm,
 const float*          A,
 float*                C,
 int                   nCols, // 0 < nCols <  bb_nCols
 int                   nRows) // nRows <= bb_nRows
{
  int lda = pPrm->lda;
  int ldc = pPrm->ldc;
  int m;
  for (m = 0; m < pPrm->M - A_WORDS_PER_ITER + 1;
    C += ldc*A_WORDS_PER_ITER,
    A += lda*A_WORDS_PER_ITER,
    m += A_WORDS_PER_ITER) {
    memcpy(&pPrm->cc[0 * B_WORDS_PER_ITER], &C[ldc * 0], nCols*sizeof(*C));
    memcpy(&pPrm->cc[1 * B_WORDS_PER_ITER], &C[ldc * 1], nCols*sizeof(*C));

    const __m256 *Bcol = pPrm->bb;
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

    Bcol += B_WORDS_PER_ITER;

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

      Bcol += B_WORDS_PER_ITER;
    }

    __m256 alpha_ps = _mm256_broadcast_ss(&pPrm->alpha);
    pPrm->cc[0 * B_WORDS_PER_ITER + 0] = _mm256_fmadd_ps(acc00, alpha_ps, pPrm->cc[0 * B_WORDS_PER_ITER + 0]);
    pPrm->cc[0 * B_WORDS_PER_ITER + 1] = _mm256_fmadd_ps(acc10, alpha_ps, pPrm->cc[0 * B_WORDS_PER_ITER + 1]);
    pPrm->cc[0 * B_WORDS_PER_ITER + 2] = _mm256_fmadd_ps(acc20, alpha_ps, pPrm->cc[0 * B_WORDS_PER_ITER + 2]);
    pPrm->cc[0 * B_WORDS_PER_ITER + 3] = _mm256_fmadd_ps(acc30, alpha_ps, pPrm->cc[0 * B_WORDS_PER_ITER + 3]);

    pPrm->cc[1 * B_WORDS_PER_ITER + 0] = _mm256_fmadd_ps(acc01, alpha_ps, pPrm->cc[1 * B_WORDS_PER_ITER + 0]);
    pPrm->cc[1 * B_WORDS_PER_ITER + 1] = _mm256_fmadd_ps(acc11, alpha_ps, pPrm->cc[1 * B_WORDS_PER_ITER + 1]);
    pPrm->cc[1 * B_WORDS_PER_ITER + 2] = _mm256_fmadd_ps(acc21, alpha_ps, pPrm->cc[1 * B_WORDS_PER_ITER + 2]);
    pPrm->cc[1 * B_WORDS_PER_ITER + 3] = _mm256_fmadd_ps(acc31, alpha_ps, pPrm->cc[1 * B_WORDS_PER_ITER + 3]);

    memcpy(&C[ldc * 0], &pPrm->cc[0 * B_WORDS_PER_ITER], nCols*sizeof(*C));
    memcpy(&C[ldc * 1], &pPrm->cc[1 * B_WORDS_PER_ITER], nCols*sizeof(*C));
  }

  if (m != pPrm->M) {
    // process a bottom row of A
    memcpy(&pPrm->cc[0], C, nCols*sizeof(*C));
    const __m256 *Bcol = pPrm->bb;
    __m256 b0 = Bcol[0];
    __m256 b1 = Bcol[1];
    __m256 b2 = Bcol[2];
    __m256 b3 = Bcol[3];
    Bcol += B_WORDS_PER_ITER;
    __m256 a;

    a = _mm256_broadcast_ss(&A[0]);
    __m256 acc00 = _mm256_mul_ps(a, b0);
    __m256 acc10 = _mm256_mul_ps(a, b1);
    __m256 acc20 = _mm256_mul_ps(a, b2);
    __m256 acc30 = _mm256_mul_ps(a, b3);

    // It is easy to better here, but it is almost certainly does not matter
    for (int k = 1; k < nRows; ++k) {
      b0 = Bcol[0];
      b1 = Bcol[1];
      b2 = Bcol[2];
      b3 = Bcol[3];
      Bcol += B_WORDS_PER_ITER;

      a = _mm256_broadcast_ss(&A[k]);
      acc00 = _mm256_fmadd_ps(a, b0, acc00);
      acc10 = _mm256_fmadd_ps(a, b1, acc10);
      acc20 = _mm256_fmadd_ps(a, b2, acc20);
      acc30 = _mm256_fmadd_ps(a, b3, acc30);
    }

    __m256 alpha_ps = _mm256_broadcast_ss(&pPrm->alpha);
    pPrm->cc[0] = _mm256_fmadd_ps(acc00, alpha_ps, pPrm->cc[0]);
    pPrm->cc[1] = _mm256_fmadd_ps(acc10, alpha_ps, pPrm->cc[1]);
    pPrm->cc[2] = _mm256_fmadd_ps(acc20, alpha_ps, pPrm->cc[2]);
    pPrm->cc[3] = _mm256_fmadd_ps(acc30, alpha_ps, pPrm->cc[3]);
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

void fma256_noncblas_sgemm_4x2(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc)
{
  noncblas_sgemm_prm_t prm;
  prm.lda   = lda;
  prm.ldc   = ldc;
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
          memcpy(&prm.bb[B_WORDS_PER_ITER*i], bSrc, bb_nCols*sizeof(*B));
          bSrc += ldb;
        }
        fma256_noncblas_sgemm_core(&prm, &Arow[row], &Crow[col]);
        col += bb_nCols;
      }
      if (col < N) {
        // process rightmost rectangle of the full-height band
        const float* bSrc = &B[row*ldb + col];
        for (int i = 0; i < bb_nRows; ++i) {
          memcpy(&prm.bb[B_WORDS_PER_ITER*i], bSrc, (N-col)*sizeof(*B));
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
          memcpy(&prm.bb[B_WORDS_PER_ITER*i], bSrc, bb_nCols*sizeof(*B));
          bSrc += ldb;
        }
        fma256_noncblas_sgemm_core_bottomRows(&prm, &Arow[row], &Crow[col], K-row);
        col += bb_nCols;
      }
      if (col < N) {
        // process bottom-right corner rectangle
        const float* bSrc = &B[row*ldb + col];
        for (int i = 0; i < K-row; ++i) {
          memcpy(&prm.bb[B_WORDS_PER_ITER*i], bSrc, (N-col)*sizeof(*B));
          bSrc += ldb;
        }
        fma256_noncblas_sgemm_core_rightmostColumns(&prm, &Arow[row], &Crow[col], N-col, K-row);
      }
    }
  }
}
