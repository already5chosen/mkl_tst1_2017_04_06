enum {
 SIMD_FACTOR             = sizeof(fp_vector_t)/sizeof(scalar_t),
 A_WORDS_PER_ITER_MJ     = 5,
 B_WORDS_PER_ITER_MJ     = 2,
 B_ITERS_PER_CORE_MAX_MJ = 3,
 SIMD_ELEM_PEC_COL_MJ    = B_WORDS_PER_ITER_MJ*B_ITERS_PER_CORE_MAX_MJ,
 bb_nCols_MJ             = SIMD_ELEM_PEC_COL_MJ*SIMD_FACTOR,
 A_WORDS_PER_ITER_MN     = 3,
 B_WORDS_PER_ITER_MN     = 3,
 B_ITERS_PER_CORE_MAX_MN = 2,
 SIMD_ELEM_PEC_COL_MN    = B_WORDS_PER_ITER_MN*B_ITERS_PER_CORE_MAX_MN,
 bb_nCols_MN             = SIMD_ELEM_PEC_COL_MN*SIMD_FACTOR,
};

struct noncblas_sgemm_prm_t {
  int          M;
  int          lda;
  int          ldc;
  scalar_t     alpha;
  int_vector_t mask[2];
  union {
    fp_vector_t  bb_maj[SIMD_ELEM_PEC_COL_MJ*bb_nRows];
    fp_vector_t  bb_min[SIMD_ELEM_PEC_COL_MN*bb_nRows];
  };
};

// major core - innner loop processes 2 SIMD columns of B x 5 rows of A
static void fma256_noncblas_sgemm_core_maj(
 const noncblas_sgemm_prm_t* pPrm,
 const scalar_t*             A,
 scalar_t*                   C,
 int                         n_bIters, // 0 < n_bIters <= B_ITERS_PER_CORE_MAX_MJ
 int                         nRows)    // 0 < nRows    <= bb_nRows
{
  int ldc = pPrm->ldc;
  int ldbb = B_WORDS_PER_ITER_MJ*nRows;
  int kSteps = unsigned(nRows-1) / 2;
  int lastRow = nRows-1-kSteps*2;
  int m;
  const int lda1 = pPrm->lda;
  const int lda2 = lda1*2;
  const int lda3 = lda1*3;
  const int lda4 = lda1*4;
  for (m = 0; m < pPrm->M-A_WORDS_PER_ITER_MJ-1; A += lda1*A_WORDS_PER_ITER_MJ, C += ldc*A_WORDS_PER_ITER_MJ, m += A_WORDS_PER_ITER_MJ) {
    scalar_t* Crow = &C[n_bIters*B_WORDS_PER_ITER_MJ*SIMD_FACTOR];
    int lastCol = 1;
    for (int b_it = n_bIters - 1; b_it >= 0; --b_it) {
      const fp_vector_t *Bcol = &pPrm->bb_maj[ldbb*b_it];
      fp_vector_t b0 = Bcol[0];
      fp_vector_t b1 = Bcol[1];
      fp_vector_t a;

      a = MM_BROADCAST_Sx(&A[0]);
      fp_vector_t acc00 = MM_MUL_Px(a, b0);
      fp_vector_t acc10 = MM_MUL_Px(a, b1);

      a = MM_BROADCAST_Sx(&A[lda1]);
      fp_vector_t acc01 = MM_MUL_Px(a, b0);
      fp_vector_t acc11 = MM_MUL_Px(a, b1);

      a = MM_BROADCAST_Sx(&A[lda2]);
      fp_vector_t acc02 = MM_MUL_Px(a, b0);
      fp_vector_t acc12 = MM_MUL_Px(a, b1);

      a = MM_BROADCAST_Sx(&A[lda3]);
      fp_vector_t acc03 = MM_MUL_Px(a, b0);
      fp_vector_t acc13 = MM_MUL_Px(a, b1);

      a = MM_BROADCAST_Sx(&A[lda4]);
      fp_vector_t acc04 = MM_MUL_Px(a, b0);
      fp_vector_t acc14 = MM_MUL_Px(a, b1);

      const scalar_t* ARow = &A[1];
      Bcol += B_WORDS_PER_ITER_MJ;
      //_mm_prefetch((char*)(Crow0 - B_WORDS_PER_ITER_MJ*SIMD_FACTOR), _MM_HINT_T0);
      //_mm_prefetch((char*)(Crow0 - B_WORDS_PER_ITER_MJ*SIMD_FACTOR + ldc), _MM_HINT_T0);
      for (int k = 0; k < kSteps; ++k) {
        b0 = Bcol[0];
        b1 = Bcol[1];
        Bcol += B_WORDS_PER_ITER_MJ;

        a = MM_BROADCAST_Sx(&ARow[0]);
        acc00 = MM_FMADD(a, b0, acc00);
        acc10 = MM_FMADD(a, b1, acc10);

        a = MM_BROADCAST_Sx(&ARow[lda1+0]);
        acc01 = MM_FMADD(a, b0, acc01);
        acc11 = MM_FMADD(a, b1, acc11);

        a = MM_BROADCAST_Sx(&ARow[lda2+0]);
        acc02 = MM_FMADD(a, b0, acc02);
        acc12 = MM_FMADD(a, b1, acc12);

        a = MM_BROADCAST_Sx(&ARow[lda3+0]);
        acc03 = MM_FMADD(a, b0, acc03);
        acc13 = MM_FMADD(a, b1, acc13);

        a = MM_BROADCAST_Sx(&ARow[lda4+0]);
        acc04 = MM_FMADD(a, b0, acc04);
        acc14 = MM_FMADD(a, b1, acc14);

        b0 = Bcol[0];
        b1 = Bcol[1];
        Bcol += B_WORDS_PER_ITER_MJ;

        a = MM_BROADCAST_Sx(&ARow[1]);
        acc00 = MM_FMADD(a, b0, acc00);
        acc10 = MM_FMADD(a, b1, acc10);

        a = MM_BROADCAST_Sx(&ARow[lda1+1]);
        acc01 = MM_FMADD(a, b0, acc01);
        acc11 = MM_FMADD(a, b1, acc11);

        a = MM_BROADCAST_Sx(&ARow[lda2+1]);
        acc02 = MM_FMADD(a, b0, acc02);
        acc12 = MM_FMADD(a, b1, acc12);

        a = MM_BROADCAST_Sx(&ARow[lda3+1]);
        acc03 = MM_FMADD(a, b0, acc03);
        acc13 = MM_FMADD(a, b1, acc13);

        a = MM_BROADCAST_Sx(&ARow[lda4+1]);
        acc04 = MM_FMADD(a, b0, acc04);
        acc14 = MM_FMADD(a, b1, acc14);
        ARow += 2;
      }

      if (lastRow) {
        b0 = Bcol[0];
        b1 = Bcol[1];

        a = MM_BROADCAST_Sx(&ARow[0]);
        acc00 = MM_FMADD(a, b0, acc00);
        acc10 = MM_FMADD(a, b1, acc10);

        a = MM_BROADCAST_Sx(&ARow[lda1+0]);
        acc01 = MM_FMADD(a, b0, acc01);
        acc11 = MM_FMADD(a, b1, acc11);

        a = MM_BROADCAST_Sx(&ARow[lda2+0]);
        acc02 = MM_FMADD(a, b0, acc02);
        acc12 = MM_FMADD(a, b1, acc12);

        a = MM_BROADCAST_Sx(&ARow[lda3+0]);
        acc03 = MM_FMADD(a, b0, acc03);
        acc13 = MM_FMADD(a, b1, acc13);

        a = MM_BROADCAST_Sx(&ARow[lda4+0]);
        acc04 = MM_FMADD(a, b0, acc04);
        acc14 = MM_FMADD(a, b1, acc14);
      }

      Crow -= B_WORDS_PER_ITER_MJ*SIMD_FACTOR;
      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
      scalar_t* Crow1 = Crow;
      int_vector_t mask = pPrm->mask[lastCol];

      MM_STOREU_Px    (&Crow1[SIMD_FACTOR*0],       MM_FMADD(acc00, alpha_ps, MM_LOADU_Px    (&Crow1[SIMD_FACTOR*0])));
      MM_MASKSTOREU_Px(&Crow1[SIMD_FACTOR*1], mask, MM_FMADD(acc10, alpha_ps, MM_MASKLOADU_Px(&Crow1[SIMD_FACTOR*1], mask)));
      Crow1 += ldc;

      MM_STOREU_Px    (&Crow1[SIMD_FACTOR*0],       MM_FMADD(acc01, alpha_ps, MM_LOADU_Px    (&Crow1[SIMD_FACTOR*0])));
      MM_MASKSTOREU_Px(&Crow1[SIMD_FACTOR*1], mask, MM_FMADD(acc11, alpha_ps, MM_MASKLOADU_Px(&Crow1[SIMD_FACTOR*1], mask)));
      Crow1 += ldc;

      MM_STOREU_Px    (&Crow1[SIMD_FACTOR*0],       MM_FMADD(acc02, alpha_ps, MM_LOADU_Px    (&Crow1[SIMD_FACTOR*0])));
      MM_MASKSTOREU_Px(&Crow1[SIMD_FACTOR*1], mask, MM_FMADD(acc12, alpha_ps, MM_MASKLOADU_Px(&Crow1[SIMD_FACTOR*1], mask)));
      Crow1 += ldc;

      MM_STOREU_Px    (&Crow1[SIMD_FACTOR*0],       MM_FMADD(acc03, alpha_ps, MM_LOADU_Px    (&Crow1[SIMD_FACTOR*0])));
      MM_MASKSTOREU_Px(&Crow1[SIMD_FACTOR*1], mask, MM_FMADD(acc13, alpha_ps, MM_MASKLOADU_Px(&Crow1[SIMD_FACTOR*1], mask)));
      Crow1 += ldc;

      MM_STOREU_Px    (&Crow1[SIMD_FACTOR*0],       MM_FMADD(acc04, alpha_ps, MM_LOADU_Px    (&Crow1[SIMD_FACTOR*0])));
      MM_MASKSTOREU_Px(&Crow1[SIMD_FACTOR*1], mask, MM_FMADD(acc14, alpha_ps, MM_MASKLOADU_Px(&Crow1[SIMD_FACTOR*1], mask)));

      lastCol = 0;
    }
  }

  for (; m < pPrm->M; A += lda1, C += ldc, ++m) {
    scalar_t* Crow0 = &C[n_bIters*B_WORDS_PER_ITER_MJ*SIMD_FACTOR];
    int lastCol = 1;
    for (int b_it = n_bIters-1; b_it >= 0; --b_it) {
      const fp_vector_t *Bcol = &pPrm->bb_maj[ldbb*b_it];
      fp_vector_t a0 = MM_BROADCAST_Sx(&A[0]);
      fp_vector_t acc00 = MM_MUL_Px(a0, Bcol[0]);
      fp_vector_t acc10 = MM_MUL_Px(a0, Bcol[1]);
      Bcol += B_WORDS_PER_ITER_MJ;
      for (int k = 1; k < nRows; ++k) {
        fp_vector_t a0 = MM_BROADCAST_Sx(&A[k]);
        acc00 = MM_FMADD(a0, Bcol[0], acc00);
        acc10 = MM_FMADD(a0, Bcol[1], acc10);
        Bcol += B_WORDS_PER_ITER_MJ;
      }
      Crow0 -= B_WORDS_PER_ITER_MJ*SIMD_FACTOR;
      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
      int_vector_t mask = pPrm->mask[lastCol];

      MM_STOREU_Px    (&Crow0[SIMD_FACTOR*0], MM_FMADD      (acc00, alpha_ps, MM_LOADU_Px    (&Crow0[SIMD_FACTOR*0])));
      MM_MASKSTOREU_Px(&Crow0[SIMD_FACTOR*1], mask, MM_FMADD(acc10, alpha_ps, MM_MASKLOADU_Px(&Crow0[SIMD_FACTOR*1], mask)));

      lastCol = 0;
    }
  }
}

static void fma256_noncblas_sgemm_core_mn(
 const noncblas_sgemm_prm_t* pPrm,
 const scalar_t*                A,
 scalar_t*                      C,
 int                         n_bIters, // 0 < n_bIters <= B_ITERS_PER_CORE_MAX_MN
 int                         nRows)    // 0 < nRows    <= bb_nRows
{
  int lda = pPrm->lda;
  int ldc = pPrm->ldc;
  int ldbb = B_WORDS_PER_ITER_MN*nRows;
  int kSteps = unsigned(nRows-1) / 2;
  int lastRow = nRows-1-kSteps*2;
  int m;
  for (m = 0; m < pPrm->M - 1; A += lda*A_WORDS_PER_ITER_MN, C += ldc*A_WORDS_PER_ITER_MN, m += A_WORDS_PER_ITER_MN) {
    scalar_t* Crow0 = &C[n_bIters*B_WORDS_PER_ITER_MN*SIMD_FACTOR];
    int lastCol = 1;
    for (int b_it = n_bIters - 1; b_it >= 0; --b_it) {
      const fp_vector_t *Bcol = &pPrm->bb_min[b_it*ldbb];
      fp_vector_t a0 = MM_BROADCAST_Sx(&A[0]);
      fp_vector_t a1 = MM_BROADCAST_Sx(&A[lda]);
      fp_vector_t b;
      b = Bcol[0];
      fp_vector_t acc00 = MM_MUL_Px(a0, b);
      fp_vector_t acc01 = MM_MUL_Px(a1, b);

      b = Bcol[1];
      fp_vector_t acc10 = MM_MUL_Px(a0, b);
      fp_vector_t acc11 = MM_MUL_Px(a1, b);

      b = Bcol[2];
      fp_vector_t acc20 = MM_MUL_Px(a0, b);
      fp_vector_t acc21 = MM_MUL_Px(a1, b);

      b = Bcol[3];
      fp_vector_t acc30 = MM_MUL_Px(a0, b);
      fp_vector_t acc31 = MM_MUL_Px(a1, b);

      const scalar_t* ARow = &A[1];
      Bcol += B_WORDS_PER_ITER_MN;
      for (int k = 0; k < kSteps; ++k) {
        a0 = MM_BROADCAST_Sx(&ARow[0]);
        a1 = MM_BROADCAST_Sx(&ARow[lda]);

        b = Bcol[0];
        acc00 = MM_FMADD(a0, b, acc00);
        acc01 = MM_FMADD(a1, b, acc01);

        b = Bcol[1];
        acc10 = MM_FMADD(a0, b, acc10);
        acc11 = MM_FMADD(a1, b, acc11);

        b = Bcol[2];
        acc20 = MM_FMADD(a0, b, acc20);
        acc21 = MM_FMADD(a1, b, acc21);

        b = Bcol[3];
        acc30 = MM_FMADD(a0, b, acc30);
        acc31 = MM_FMADD(a1, b, acc31);

        Bcol += B_WORDS_PER_ITER_MN;
        a0 = MM_BROADCAST_Sx(&ARow[1]);
        a1 = MM_BROADCAST_Sx(&ARow[lda + 1]);

        b = Bcol[0];
        acc00 = MM_FMADD(a0, b, acc00);
        acc01 = MM_FMADD(a1, b, acc01);

        b = Bcol[1];
        acc10 = MM_FMADD(a0, b, acc10);
        acc11 = MM_FMADD(a1, b, acc11);

        b = Bcol[2];
        acc20 = MM_FMADD(a0, b, acc20);
        acc21 = MM_FMADD(a1, b, acc21);

        b = Bcol[3];
        acc30 = MM_FMADD(a0, b, acc30);
        acc31 = MM_FMADD(a1, b, acc31);

        ARow += 2;
        Bcol += B_WORDS_PER_ITER_MN;
      }

      if (lastRow) {
        a0 = MM_BROADCAST_Sx(&ARow[0]);
        a1 = MM_BROADCAST_Sx(&ARow[lda + 0]);

        b = Bcol[0];
        acc00 = MM_FMADD(a0, b, acc00);
        acc01 = MM_FMADD(a1, b, acc01);

        b = Bcol[1];
        acc10 = MM_FMADD(a0, b, acc10);
        acc11 = MM_FMADD(a1, b, acc11);

        b = Bcol[2];
        acc20 = MM_FMADD(a0, b, acc20);
        acc21 = MM_FMADD(a1, b, acc21);

        b = Bcol[3];
        acc30 = MM_FMADD(a0, b, acc30);
        acc31 = MM_FMADD(a1, b, acc31);
      }

      Crow0 -= B_WORDS_PER_ITER_MN*SIMD_FACTOR;
      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
      int_vector_t mask = pPrm->mask[lastCol];

      MM_STOREU_Px(&Crow0[SIMD_FACTOR * 0], MM_FMADD(acc00, alpha_ps, MM_LOADU_Px(&Crow0[SIMD_FACTOR * 0])));
      MM_STOREU_Px(&Crow0[SIMD_FACTOR * 1], MM_FMADD(acc10, alpha_ps, MM_LOADU_Px(&Crow0[SIMD_FACTOR * 1])));
      MM_STOREU_Px(&Crow0[SIMD_FACTOR * 2], MM_FMADD(acc20, alpha_ps, MM_LOADU_Px(&Crow0[SIMD_FACTOR * 2])));
      MM_MASKSTOREU_Px(&Crow0[SIMD_FACTOR * 3], mask, MM_FMADD(acc30, alpha_ps, MM_MASKLOADU_Px(&Crow0[SIMD_FACTOR * 3], mask)));

      scalar_t* Crow1 = Crow0 + ldc;
      MM_STOREU_Px(&Crow1[SIMD_FACTOR * 0], MM_FMADD(acc01, alpha_ps, MM_LOADU_Px(&Crow1[SIMD_FACTOR * 0])));
      MM_STOREU_Px(&Crow1[SIMD_FACTOR * 1], MM_FMADD(acc11, alpha_ps, MM_LOADU_Px(&Crow1[SIMD_FACTOR * 1])));
      MM_STOREU_Px(&Crow1[SIMD_FACTOR * 2], MM_FMADD(acc21, alpha_ps, MM_LOADU_Px(&Crow1[SIMD_FACTOR * 2])));
      MM_MASKSTOREU_Px(&Crow1[SIMD_FACTOR * 3], mask, MM_FMADD(acc31, alpha_ps, MM_MASKLOADU_Px(&Crow1[SIMD_FACTOR * 3], mask)));

      lastCol = 0;
    }
  }
  if (m < pPrm->M) {
    scalar_t* Crow0 = &C[n_bIters*B_WORDS_PER_ITER_MN*SIMD_FACTOR];
    int lastCol = 1;
    for (int b_it = n_bIters - 1; b_it >= 0; --b_it) {
      const fp_vector_t *Bcol = &pPrm->bb_min[b_it*ldbb];
      fp_vector_t a0 = MM_BROADCAST_Sx(&A[0]);
      fp_vector_t acc00 = MM_MUL_Px(a0, Bcol[0]);
      fp_vector_t acc10 = MM_MUL_Px(a0, Bcol[1]);
      fp_vector_t acc20 = MM_MUL_Px(a0, Bcol[2]);
      fp_vector_t acc30 = MM_MUL_Px(a0, Bcol[3]);
      Bcol += B_WORDS_PER_ITER_MN;
      for (int k = 1; k < nRows; ++k) {
        fp_vector_t a0 = MM_BROADCAST_Sx(&A[k]);
        acc00 = MM_FMADD(a0, Bcol[0], acc00);
        acc10 = MM_FMADD(a0, Bcol[1], acc10);
        acc20 = MM_FMADD(a0, Bcol[2], acc20);
        acc30 = MM_FMADD(a0, Bcol[3], acc30);
        Bcol += B_WORDS_PER_ITER_MN;
      }
      Crow0 -= B_WORDS_PER_ITER_MN*SIMD_FACTOR;
      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);

      int_vector_t mask = pPrm->mask[lastCol];
      MM_STOREU_Px(&Crow0[SIMD_FACTOR * 0], MM_FMADD(acc00, alpha_ps, MM_LOADU_Px(&Crow0[SIMD_FACTOR * 0])));
      MM_STOREU_Px(&Crow0[SIMD_FACTOR * 1], MM_FMADD(acc10, alpha_ps, MM_LOADU_Px(&Crow0[SIMD_FACTOR * 1])));
      MM_STOREU_Px(&Crow0[SIMD_FACTOR * 2], MM_FMADD(acc20, alpha_ps, MM_LOADU_Px(&Crow0[SIMD_FACTOR * 2])));
      MM_MASKSTOREU_Px(&Crow0[SIMD_FACTOR * 3], mask, MM_FMADD(acc30, alpha_ps, MM_MASKLOADU_Px(&Crow0[SIMD_FACTOR * 3], mask)));

      lastCol = 0;
    }
  }
}


static void fma256_noncblas_sgemm_multC(
 int M, int N,
 scalar_t beta,
 scalar_t *C, int ldc)
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

static void CopyAndTransposeMj(
  noncblas_sgemm_prm_t* pPrm,
  const scalar_t *B, int ldb,
  int n_bIters, int nRows)
{
  fp_vector_t* dstCol = pPrm->bb_maj;
  int ldbb = nRows*B_WORDS_PER_ITER_MJ;
  for (int r = 0; r < nRows; ++r) {
    const scalar_t *src = B;
    fp_vector_t* dst = dstCol;
    for (int c = 0; c < n_bIters; ++c) {
      memcpy(dst, src, sizeof(*dst)*B_WORDS_PER_ITER_MJ);
      src += SIMD_FACTOR*B_WORDS_PER_ITER_MJ;
      dst += ldbb;
    }
    B      += ldb;
    dstCol += B_WORDS_PER_ITER_MJ;
  }
}

static void CopyAndTransposeMjWithMask(
  noncblas_sgemm_prm_t* pPrm,
  const scalar_t *B, int ldb,
  int n_bIters, int nRows)
{
  fp_vector_t* dstCol = pPrm->bb_maj;
  int ldbb = nRows*B_WORDS_PER_ITER_MJ;
  int_vector_t mask = pPrm->mask[1];
  for (int r = 0; r < nRows; ++r) {
    const scalar_t *src = B;
    fp_vector_t* dst = dstCol;
    for (int c = 0; c < n_bIters-1; ++c) {
      memcpy(dst, src, sizeof(*dst)*B_WORDS_PER_ITER_MJ);
      src += SIMD_FACTOR*B_WORDS_PER_ITER_MJ;
      dst += ldbb;
    }
    memcpy(dst, src, sizeof(*dst)*(B_WORDS_PER_ITER_MJ-1));
    dst[B_WORDS_PER_ITER_MJ-1] = MM_MASKLOADU_Px(&src[(B_WORDS_PER_ITER_MJ-1)*SIMD_FACTOR], mask);
    B      += ldb;
    dstCol += B_WORDS_PER_ITER_MJ;
  }
}

static void CopyAndTransposeMnWithMask(
  noncblas_sgemm_prm_t* pPrm,
  const scalar_t *B, int ldb,
  int n_bIters, int nRows)
{
  fp_vector_t* dstCol = pPrm->bb_maj;
  int ldbb = nRows*B_WORDS_PER_ITER_MN;
  int_vector_t mask = pPrm->mask[1];
  for (int r = 0; r < nRows; ++r) {
    const scalar_t *src = B;
    fp_vector_t* dst = dstCol;
    for (int c = 0; c < n_bIters-1; ++c) {
      memcpy(dst, src, sizeof(*dst)*B_WORDS_PER_ITER_MN);
      src += SIMD_FACTOR*B_WORDS_PER_ITER_MN;
      dst += ldbb;
    }
    memcpy(dst, src, sizeof(*dst)*(B_WORDS_PER_ITER_MN-1));
    dst[B_WORDS_PER_ITER_MN-1] = MM_MASKLOADU_Px(&src[(B_WORDS_PER_ITER_MN-1)*SIMD_FACTOR], mask);
    B      += ldb;
    dstCol += B_WORDS_PER_ITER_MN;
  }
}


// N>0. ceil(N/SIMD_FACTOR) is representable as x*2+y*3 where x and y are non-negative
static void noncblas_sgemm_wide_n(
  int M, int N, int K,
  scalar_t alpha,
  const scalar_t *A, int lda,
  const scalar_t *B, int ldb,
  scalar_t beta,
  scalar_t *C, int ldc)
{
  unsigned nw = unsigned((N-1)/SIMD_FACTOR+1);
  int nMaj = nw / SIMD_ELEM_PEC_COL_MJ;
  int nRem = nw - nMaj*SIMD_ELEM_PEC_COL_MJ;
  static uint8_t remSplitTab[6][2] = {
    {0,     6/2}, // 0 => 6
    {3/3,   4/2}, // 1 => 7
    {0/3,   2/2}, // 2
    {3/3,   0/2}, // 3
    {0/3,   4/2}, // 4
    {3/3,   2/2}, // 5
  };
  int nRemMin = remSplitTab[nRem][0];
  int nRemMaj = remSplitTab[nRem][1];
  int nRem1 = nRemMin*B_WORDS_PER_ITER_MN + nRemMaj*B_WORDS_PER_ITER_MJ;
  nMaj -= (nRem1-nRem)/SIMD_ELEM_PEC_COL_MJ;

  noncblas_sgemm_prm_t prm;
  prm.lda = lda;
  prm.ldc = ldc;
  prm.alpha = alpha;
  memset(prm.mask, -1, sizeof(prm.mask[0]));
  int_vector_t lastWMask[1];
  lastWMask[0] = prm.mask[0];
  unsigned remW = nw*SIMD_FACTOR - N;
  if (remW > 0) {
    // mask off elements of rightmost SIMD word in B and C
    memset((char*)&lastWMask[1] - sizeof(*C)*remW, 0, sizeof(*C)*remW);
  }

  for (int m = 0; m < M; m += prm.M) {
    prm.M = M - m <= m_step_max ? M - m : m_step_nom;

    scalar_t *Crow = &C[m*ldc];
    fma256_noncblas_sgemm_multC(prm.M, N, beta, Crow, ldc);

    const scalar_t *Arow = &A[m*lda];
    for (int row = 0; row < K; row += bb_nRows) {
      int nRows = std::min(int(bb_nRows), K-row);
      int col = 0;
      prm.mask[1] = prm.mask[0]; // all words in use
      for (int ci = 0; ci < nMaj; ++ci) {
        // process full major rectangles
        if (row==0 && col==0)
        CopyAndTransposeMj(&prm, &B[row*ldb + col], ldb, B_ITERS_PER_CORE_MAX_MJ, nRows);
        fma256_noncblas_sgemm_core_maj(&prm, &Arow[row], &Crow[col], B_ITERS_PER_CORE_MAX_MJ, nRows);
        col += bb_nCols_MJ;
      }
      if (nRemMin > 0) {
        if (nRemMaj == 0)
          prm.mask[1] = lastWMask[0]; // mask for leftmost word
        //CopyAndTransposeMnWithMask(&prm, &B[row*ldb + col], ldb, nRemMin, nRows);
        fma256_noncblas_sgemm_core_mn(&prm, &Arow[row], &Crow[col], nRemMin, nRows);
        col += nRemMin*B_WORDS_PER_ITER_MN*SIMD_FACTOR;
      }
      if (nRemMaj > 0) {
        prm.mask[1] = lastWMask[0]; // mask for leftmost word
        //CopyAndTransposeMjWithMask(&prm, &B[row*ldb + col], ldb, nRemMaj, nRows);
        fma256_noncblas_sgemm_core_maj(&prm, &Arow[row], &Crow[col], nRemMaj, nRows);
      }
    }
  }
}

static void noncblas_sgemm_narrow_n(
  int M, int N, int K,
  scalar_t alpha,
  const scalar_t *A, int lda,
  const scalar_t *B, int ldb,
  scalar_t beta,
  scalar_t *C, int ldc)
{
  // TODO
}


void func_name(
  int M, int N, int K,
  scalar_t alpha,
  const scalar_t *A, int lda,
  const scalar_t *B, int ldb,
  scalar_t beta,
  scalar_t *C, int ldc)
{
  if (N >= SIMD_FACTOR+1) {
    noncblas_sgemm_wide_n(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  } else if (N >= 1) {
    noncblas_sgemm_narrow_n(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }
}
