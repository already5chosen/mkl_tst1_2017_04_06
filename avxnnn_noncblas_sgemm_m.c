enum {
 SIMD_FACTOR             = sizeof(fp_vector_t)/sizeof(scalar_t),
 A_WORDS_PER_ITER_MJ     = 2,
 B_WORDS_PER_ITER_MJ     = 5,
 B_ITERS_PER_CORE_MAX_MJ = 3,
 SIMD_ELEM_PEC_COL_MJ    = B_WORDS_PER_ITER_MJ*B_ITERS_PER_CORE_MAX_MJ,
 n_step                  = SIMD_ELEM_PEC_COL_MJ*SIMD_FACTOR,
 A_WORDS_PER_ITER_MN     = 2,
 B_WORDS_PER_ITER_MN     = 4,
 B_ITERS_PER_CORE_MAX_MN = 4,
 SIMD_ELEM_PEC_COL_MN    = B_WORDS_PER_ITER_MN*B_ITERS_PER_CORE_MAX_MN,
 bb_nCols_MN             = SIMD_ELEM_PEC_COL_MN*SIMD_FACTOR,
 bb_nRows                = k_step,
};

typedef struct {
  int          M;
  int          lda;
  int          ldc;
  scalar_t     alpha;
  int_vector_t mask_n[3]; // [0]= all 1s, [1] = all ones or last SIMD word, [2] = last SIMD word
  union {
    fp_vector_t  bb_maj[SIMD_ELEM_PEC_COL_MJ*bb_nRows];
    fp_vector_t  bb_min[SIMD_ELEM_PEC_COL_MN*bb_nRows];
  };
} noncblas_sgemm_prm_t;

static void fma256_noncblas_sgemm_core_mj(
 const noncblas_sgemm_prm_t* pPrm,
 const scalar_t*             A,
 scalar_t*                   C,
 int                         n_bIters, // 0 < n_bIters <= B_ITERS_PER_CORE_MAX_MJ
 int                         nRows)    // 0 < nRows    <= bb_nRows
{
  int lda = pPrm->lda;
  int ldc = pPrm->ldc;
  int kSteps = (unsigned)(nRows - 1) / 2;
  int lastRow = nRows - 1 - kSteps * 2;
  int m;
  for (m = 0; m < pPrm->M - 1; A += lda*A_WORDS_PER_ITER_MJ, C += ldc*A_WORDS_PER_ITER_MJ, m += A_WORDS_PER_ITER_MJ) {
    scalar_t* Crow0 = &C[n_bIters*B_WORDS_PER_ITER_MJ*SIMD_FACTOR];
    int lastCol = 1;
    for (int b_it = n_bIters - 1; b_it >= 0; --b_it) {
      const fp_vector_t *Bcol = &pPrm->bb_maj[b_it*B_WORDS_PER_ITER_MJ];
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

      b = Bcol[4];
      fp_vector_t acc40 = MM_MUL_Px(a0, b);
      fp_vector_t acc41 = MM_MUL_Px(a1, b);

      const scalar_t* ARow = &A[1];
      Bcol += SIMD_ELEM_PEC_COL_MJ;

      Crow0 -= B_WORDS_PER_ITER_MJ*SIMD_FACTOR;
      for (int ci = 0; ci < B_WORDS_PER_ITER_MJ; ++ci) {
        _mm_prefetch((char*)(Crow0 + ci*SIMD_FACTOR), _MM_HINT_T0);
        _mm_prefetch((char*)(Crow0 + ldc + ci*SIMD_FACTOR), _MM_HINT_T0);
      }

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

        b = Bcol[4];
        acc40 = MM_FMADD(a0, b, acc40);
        acc41 = MM_FMADD(a1, b, acc41);

        Bcol += SIMD_ELEM_PEC_COL_MJ;
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

        b = Bcol[4];
        acc40 = MM_FMADD(a0, b, acc40);
        acc41 = MM_FMADD(a1, b, acc41);
        ARow += 2;
        Bcol += SIMD_ELEM_PEC_COL_MJ;
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

        b = Bcol[4];
        acc40 = MM_FMADD(a0, b, acc40);
        acc41 = MM_FMADD(a1, b, acc41);
      }

      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);

      MM_STOREU_Px(&Crow0[SIMD_FACTOR * 0], MM_FMADD(acc00, alpha_ps, MM_LOADU_Px(&Crow0[SIMD_FACTOR * 0])));
      MM_STOREU_Px(&Crow0[SIMD_FACTOR * 1], MM_FMADD(acc10, alpha_ps, MM_LOADU_Px(&Crow0[SIMD_FACTOR * 1])));
      MM_STOREU_Px(&Crow0[SIMD_FACTOR * 2], MM_FMADD(acc20, alpha_ps, MM_LOADU_Px(&Crow0[SIMD_FACTOR * 2])));
      MM_STOREU_Px(&Crow0[SIMD_FACTOR * 3], MM_FMADD(acc30, alpha_ps, MM_LOADU_Px(&Crow0[SIMD_FACTOR * 3])));
      int_vector_t mask = pPrm->mask_n[lastCol];
      MM_MASKSTOREU_Px(&Crow0[SIMD_FACTOR * 4], mask, MM_FMADD(acc40, alpha_ps, MM_MASKLOADU_Px(&Crow0[SIMD_FACTOR * 4], mask)));

      scalar_t* Crow1 = Crow0 + ldc;
      MM_STOREU_Px(&Crow1[SIMD_FACTOR * 0], MM_FMADD(acc01, alpha_ps, MM_LOADU_Px(&Crow1[SIMD_FACTOR * 0])));
      MM_STOREU_Px(&Crow1[SIMD_FACTOR * 1], MM_FMADD(acc11, alpha_ps, MM_LOADU_Px(&Crow1[SIMD_FACTOR * 1])));
      MM_STOREU_Px(&Crow1[SIMD_FACTOR * 2], MM_FMADD(acc21, alpha_ps, MM_LOADU_Px(&Crow1[SIMD_FACTOR * 2])));
      MM_STOREU_Px(&Crow1[SIMD_FACTOR * 3], MM_FMADD(acc31, alpha_ps, MM_LOADU_Px(&Crow1[SIMD_FACTOR * 3])));
      MM_MASKSTOREU_Px(&Crow1[SIMD_FACTOR * 4], mask, MM_FMADD(acc41, alpha_ps, MM_MASKLOADU_Px(&Crow1[SIMD_FACTOR * 4], mask)));

      lastCol = 0;
    }
  }
  if (m < pPrm->M) {
    scalar_t* Crow0 = &C[n_bIters*B_WORDS_PER_ITER_MJ*SIMD_FACTOR];
    int lastCol = 1;
    for (int b_it = n_bIters - 1; b_it >= 0; --b_it) {
      const fp_vector_t *Bcol = &pPrm->bb_maj[b_it*B_WORDS_PER_ITER_MJ];
      fp_vector_t a0 = MM_BROADCAST_Sx(&A[0]);
      fp_vector_t acc00 = MM_MUL_Px(a0, Bcol[0]);
      fp_vector_t acc10 = MM_MUL_Px(a0, Bcol[1]);
      fp_vector_t acc20 = MM_MUL_Px(a0, Bcol[2]);
      fp_vector_t acc30 = MM_MUL_Px(a0, Bcol[3]);
      fp_vector_t acc40 = MM_MUL_Px(a0, Bcol[4]);

      Crow0 -= B_WORDS_PER_ITER_MJ*SIMD_FACTOR;
      for (int ci = 0; ci < B_WORDS_PER_ITER_MJ; ++ci)
        _mm_prefetch((char*)(Crow0 + ci*SIMD_FACTOR), _MM_HINT_T0);

      Bcol += SIMD_ELEM_PEC_COL_MJ;
      for (int k = 1; k < nRows; ++k) {
        fp_vector_t a0 = MM_BROADCAST_Sx(&A[k]);
        acc00 = MM_FMADD(a0, Bcol[0], acc00);
        acc10 = MM_FMADD(a0, Bcol[1], acc10);
        acc20 = MM_FMADD(a0, Bcol[2], acc20);
        acc30 = MM_FMADD(a0, Bcol[3], acc30);
        acc40 = MM_FMADD(a0, Bcol[4], acc40);
        Bcol += SIMD_ELEM_PEC_COL_MJ;
      }
      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);

      MM_STOREU_Px(&Crow0[SIMD_FACTOR * 0], MM_FMADD(acc00, alpha_ps, MM_LOADU_Px(&Crow0[SIMD_FACTOR * 0])));
      MM_STOREU_Px(&Crow0[SIMD_FACTOR * 1], MM_FMADD(acc10, alpha_ps, MM_LOADU_Px(&Crow0[SIMD_FACTOR * 1])));
      MM_STOREU_Px(&Crow0[SIMD_FACTOR * 2], MM_FMADD(acc20, alpha_ps, MM_LOADU_Px(&Crow0[SIMD_FACTOR * 2])));
      MM_STOREU_Px(&Crow0[SIMD_FACTOR * 3], MM_FMADD(acc30, alpha_ps, MM_LOADU_Px(&Crow0[SIMD_FACTOR * 3])));
      int_vector_t mask = pPrm->mask_n[lastCol];
      MM_MASKSTOREU_Px(&Crow0[SIMD_FACTOR * 4], mask, MM_FMADD(acc40, alpha_ps, MM_MASKLOADU_Px(&Crow0[SIMD_FACTOR * 4], mask)));

      lastCol = 0;
    }
  }
}

static void fma256_noncblas_sgemm_core_mn(
 const noncblas_sgemm_prm_t* pPrm,
 const scalar_t*             A,
 scalar_t*                   C,
 int                         n_bIters, // 0 < n_bIters <= B_ITERS_PER_CORE_MAX_MN
 int                         nRows)    // 0 < nRows    <= bb_nRows
{
  int lda = pPrm->lda;
  int ldc = pPrm->ldc;
  int kSteps = (unsigned)(nRows - 1) / 2;
  int lastRow = nRows - 1 - kSteps * 2;
  int m;
  for (m = 0; m < pPrm->M - 1; A += lda*A_WORDS_PER_ITER_MN, C += ldc*A_WORDS_PER_ITER_MN, m += A_WORDS_PER_ITER_MN) {
    scalar_t* Crow0 = &C[n_bIters*B_WORDS_PER_ITER_MN*SIMD_FACTOR];
    int lastCol = 1;
    for (int b_it = n_bIters - 1; b_it >= 0; --b_it) {
      const fp_vector_t *Bcol = &pPrm->bb_min[b_it*B_WORDS_PER_ITER_MN];
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

      Crow0 -= B_WORDS_PER_ITER_MN*SIMD_FACTOR;
      for (int ci = 0; ci < B_WORDS_PER_ITER_MN; ++ci) {
        _mm_prefetch((char*)(Crow0 + ci*SIMD_FACTOR), _MM_HINT_T0);
        _mm_prefetch((char*)(Crow0 + ldc + ci*SIMD_FACTOR), _MM_HINT_T0);
      }

      const scalar_t* ARow = &A[1];
      Bcol += SIMD_ELEM_PEC_COL_MN;
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

        Bcol += SIMD_ELEM_PEC_COL_MN;
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
        Bcol += SIMD_ELEM_PEC_COL_MN;
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

      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
      int_vector_t mask = pPrm->mask_n[lastCol];

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
      Crow0 -= B_WORDS_PER_ITER_MN*SIMD_FACTOR;
      for (int ci = 0; ci < B_WORDS_PER_ITER_MN; ++ci)
        _mm_prefetch((char*)(Crow0 + ci*SIMD_FACTOR), _MM_HINT_T0);

      const fp_vector_t *Bcol = &pPrm->bb_min[b_it*B_WORDS_PER_ITER_MN];
      fp_vector_t a0 = MM_BROADCAST_Sx(&A[0]);
      fp_vector_t acc00 = MM_MUL_Px(a0, Bcol[0]);
      fp_vector_t acc10 = MM_MUL_Px(a0, Bcol[1]);
      fp_vector_t acc20 = MM_MUL_Px(a0, Bcol[2]);
      fp_vector_t acc30 = MM_MUL_Px(a0, Bcol[3]);
      Bcol += SIMD_ELEM_PEC_COL_MN;
      for (int k = 1; k < nRows; ++k) {
        fp_vector_t a0 = MM_BROADCAST_Sx(&A[k]);
        acc00 = MM_FMADD(a0, Bcol[0], acc00);
        acc10 = MM_FMADD(a0, Bcol[1], acc10);
        acc20 = MM_FMADD(a0, Bcol[2], acc20);
        acc30 = MM_FMADD(a0, Bcol[3], acc30);
        Bcol += SIMD_ELEM_PEC_COL_MN;
      }
      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);

      int_vector_t mask = pPrm->mask_n[lastCol];
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

static void CopyBWithMask(noncblas_sgemm_prm_t* pPrm, int ldbb, const scalar_t *B, int ldb, int nCol, int nRows)
{
  fp_vector_t* dst = pPrm->bb_maj;
  int_vector_t mask = pPrm->mask_n[1];
  for (int r = 0; r < nRows; ++r) {
    for (int c = 0; c < nCol-1; ++c)
      dst[c] = MM_LOADU_Px(&B[c*SIMD_FACTOR]);
    dst[nCol-1] = MM_MASKLOADU_Px(&B[(nCol-1)*SIMD_FACTOR], mask);
    B   += ldb;
    dst += ldbb;
  }
}

// N>0. ceil(N/SIMD_FACTOR) is representable as x*5+y*4 where x and y are non-negative
static void noncblas_sgemm_wide_n(
  int M, int N, int K,
  scalar_t alpha,
  const scalar_t *A, int lda,
  const scalar_t *B, int ldb,
  scalar_t beta,
  scalar_t *C, int ldc)
{
  unsigned nw = (unsigned)(N-1)/SIMD_FACTOR+1;
  int nwMj  = nw / SIMD_ELEM_PEC_COL_MJ;
  int nwRem = nw - nwMj*SIMD_ELEM_PEC_COL_MJ;
  static uint8_t remSplitTab[15][2] = {
    {0,    15/5}, // 0 => 15
    {16/4,  0/5}, // 1 => 16
    {12/4,  5/5}, // 2 => 17
    { 8/4, 10/5}, // 3 => 18
    { 4/4,  0/5}, // 4
    { 0/4,  5/5}, // 5
    {16/4,  5/5}, // 6 => 21
    {12/4, 10/5}, // 7 => 22
    { 8/4,  0/5}, // 8
    { 4/4,  5/5}, // 9
    { 0/4, 10/5}, // 10
    {16/4, 10/5}, // 11=> 26
    {12/4,  0/5}, // 12
    { 8/4,  5/5}, // 13
    { 4/4, 10/5}, // 14
  };
  int nwRemMn = remSplitTab[nwRem][0];
  int nwRemMj = remSplitTab[nwRem][1];
  int nMj = (nw - nwRemMn*B_WORDS_PER_ITER_MN - nwRemMj*B_WORDS_PER_ITER_MJ) * SIMD_FACTOR;

  noncblas_sgemm_prm_t prm;
  prm.lda = lda;
  prm.ldc = ldc;
  prm.alpha = alpha;
  memset(&prm.mask_n[0], -1, sizeof(prm.mask_n[0]));
  prm.mask_n[2] = prm.mask_n[0];
  unsigned remW = nw*SIMD_FACTOR - N;
  if (remW > 0) // mask off elements of rightmost SIMD word in B and C
    memset((char*)&prm.mask_n[3] - sizeof(*C)*remW, 0, sizeof(*C)*remW);

  for (int m = 0; m < M; m += prm.M) {
    prm.M = M - m <= m_step_max ? M - m : m_step_nom;

    scalar_t *Crow = &C[m*ldc];
    fma256_noncblas_sgemm_multC(prm.M, N, beta, Crow, ldc);

    const scalar_t *Arow = &A[m*lda];
    for (int k = 0; k < K; k += k_step) {
      int delta_k = k_step < K-k ? k_step : K-k;
      prm.mask_n[1] = prm.mask_n[0]; // all words in use
      int n;
      for (n = 0; n < nMj; n += n_step) {
        // process full major rectangles
        const scalar_t* bSrc = &B[k*ldb + n];
        for (int i = 0; i < delta_k; ++i) {
          memcpy(&prm.bb_maj[SIMD_ELEM_PEC_COL_MJ*i], bSrc, n_step*sizeof(*B));
          bSrc += ldb;
        }
        fma256_noncblas_sgemm_core_mj(&prm, &Arow[k], &Crow[n], B_ITERS_PER_CORE_MAX_MJ, delta_k);
      }
      if (nwRemMn > 0) {
        if (nwRemMj == 0)
          prm.mask_n[1] = prm.mask_n[2]; // mask for leftmost word
        CopyBWithMask(&prm, SIMD_ELEM_PEC_COL_MN, &B[k*ldb + n], ldb, nwRemMn*B_WORDS_PER_ITER_MN, delta_k);
        fma256_noncblas_sgemm_core_mn(&prm, &Arow[k], &Crow[n], nwRemMn, delta_k);
        n += nwRemMn*B_WORDS_PER_ITER_MN*SIMD_FACTOR;
      }
      if (nwRemMj > 0) {
        prm.mask_n[1] = prm.mask_n[2]; // mask for leftmost word
        CopyBWithMask(&prm, SIMD_ELEM_PEC_COL_MJ, &B[k*ldb + n], ldb, nwRemMj*B_WORDS_PER_ITER_MJ, delta_k);
        fma256_noncblas_sgemm_core_mj(&prm, &Arow[k], &Crow[n], nwRemMj, delta_k);
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
  if (N >= SIMD_FACTOR*11+1) {
    noncblas_sgemm_wide_n(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  } else if (N >= 1) {
    unsigned nw = (unsigned)(N-1)/SIMD_FACTOR + 1;
    switch (nw) {
      case 4:
      case 5:
      case 8:
      case 9:
      case 10:
        noncblas_sgemm_wide_n(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        break;
      default:
        noncblas_sgemm_narrow_n(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        break;
    }
  }
}
