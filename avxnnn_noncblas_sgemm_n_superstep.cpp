enum {
 SIMD_FACTOR          = sizeof(fp_vector_t)/sizeof(scalar_t),
 A_WORDS_PER_ITER     = 5,
 B_WORDS_PER_ITER     = 2,
 SIMD_ELEM_PEC_COL_MJ = B_WORDS_PER_ITER*N_STEP_MULTIPLIER,
 n_step               = SIMD_ELEM_PEC_COL_MJ*SIMD_FACTOR,
};

struct noncblas_sgemm_prm_t {
  int           M;
  int           lda;
  int           ldc;
  scalar_t      alpha;
  int_vector4_t mask_a[2]; // [0]= all 1s, [2] = last vector4 word
  int_vector_t  mask_b[3]; // [0]= all 1s, [1] = all ones or last SIMD word, [2] = last SIMD word
  // fp_vector_t   bb[SIMD_ELEM_PEC_COL_MJ*k_step];
  // fp_vector4_t  aa[(k_step*m_step_max)/4];
  fp_vector_t*  bb; // [SIMD_ELEM_PEC_COL_MJ*k_step];
  fp_vector4_t* aa; // [(k_step*m_step_max)/4];
};

// major core - inner loop processes 2 SIMD columns of B x 5 rows of A
static void fma256_noncblas_sgemm_core_mj(
 const noncblas_sgemm_prm_t* pPrm,
 scalar_t*                   C,
 int                         n_bIters, // 0 < n_bIters <= N_STEP_MULTIPLIER
 int                         nRows)    // 0 < nRows    <= k_step
{
  int ldc = pPrm->ldc;
  int kSteps = unsigned(nRows-1) / 4 + 1;
  int ldbb   = B_WORDS_PER_ITER*nRows;
  int m;
  const fp_vector4_t* A = pPrm->aa;
  for (m = 0; m < pPrm->M-A_WORDS_PER_ITER+1;
    A += kSteps*A_WORDS_PER_ITER,
    C += ldc*A_WORDS_PER_ITER,
    m += A_WORDS_PER_ITER) {
    scalar_t* Crow = &C[n_bIters*B_WORDS_PER_ITER*SIMD_FACTOR];
    int lastCol = 1;
    for (int b_it = n_bIters - 1; b_it >= 0; --b_it) {
      const fp_vector_t* Bcol = &pPrm->bb[ldbb*b_it];
      const scalar_t*    ARow = reinterpret_cast<const scalar_t*>(A);
      fp_vector_t acc00 = MM_SETZERO_Px();
      fp_vector_t acc10 = MM_SETZERO_Px();
      fp_vector_t acc01 = MM_SETZERO_Px();
      fp_vector_t acc11 = MM_SETZERO_Px();
      fp_vector_t acc02 = MM_SETZERO_Px();
      fp_vector_t acc12 = MM_SETZERO_Px();
      fp_vector_t acc03 = MM_SETZERO_Px();
      fp_vector_t acc13 = MM_SETZERO_Px();
      fp_vector_t acc04 = MM_SETZERO_Px();
      fp_vector_t acc14 = MM_SETZERO_Px();

      Crow -= B_WORDS_PER_ITER*SIMD_FACTOR;
      scalar_t* CCol = Crow;
      for (int ci = 0; ci < A_WORDS_PER_ITER; ++ci) {
        _mm_prefetch((char*)(CCol),             _MM_HINT_T0);
        _mm_prefetch((char*)(CCol+SIMD_FACTOR), _MM_HINT_T0);
        CCol += ldc;
      }

      for (int k = 0; k < kSteps; ++k) {
        fp_vector_t a, b0, b1;
        b0 = Bcol[0];
        b1 = Bcol[1];
        Bcol += B_WORDS_PER_ITER;

        a = MM_BROADCAST_Sx(&ARow[4*0+0]);
        acc00 = MM_FMADD(a, b0, acc00);
        acc10 = MM_FMADD(a, b1, acc10);

        a = MM_BROADCAST_Sx(&ARow[4*1+0]);
        acc01 = MM_FMADD(a, b0, acc01);
        acc11 = MM_FMADD(a, b1, acc11);

        a = MM_BROADCAST_Sx(&ARow[4*2+0]);
        acc02 = MM_FMADD(a, b0, acc02);
        acc12 = MM_FMADD(a, b1, acc12);

        a = MM_BROADCAST_Sx(&ARow[4*3+0]);
        acc03 = MM_FMADD(a, b0, acc03);
        acc13 = MM_FMADD(a, b1, acc13);

        a = MM_BROADCAST_Sx(&ARow[4*4+0]);
        acc04 = MM_FMADD(a, b0, acc04);
        acc14 = MM_FMADD(a, b1, acc14);

        b0 = Bcol[0];
        b1 = Bcol[1];
        Bcol += B_WORDS_PER_ITER;

        a = MM_BROADCAST_Sx(&ARow[4*0+1]);
        acc00 = MM_FMADD(a, b0, acc00);
        acc10 = MM_FMADD(a, b1, acc10);

        a = MM_BROADCAST_Sx(&ARow[4*1+1]);
        acc01 = MM_FMADD(a, b0, acc01);
        acc11 = MM_FMADD(a, b1, acc11);

        a = MM_BROADCAST_Sx(&ARow[4*2+1]);
        acc02 = MM_FMADD(a, b0, acc02);
        acc12 = MM_FMADD(a, b1, acc12);

        a = MM_BROADCAST_Sx(&ARow[4*3+1]);
        acc03 = MM_FMADD(a, b0, acc03);
        acc13 = MM_FMADD(a, b1, acc13);

        a = MM_BROADCAST_Sx(&ARow[4*4+1]);
        acc04 = MM_FMADD(a, b0, acc04);
        acc14 = MM_FMADD(a, b1, acc14);

        b0 = Bcol[0];
        b1 = Bcol[1];
        Bcol += B_WORDS_PER_ITER;

        a = MM_BROADCAST_Sx(&ARow[4*0+2]);
        acc00 = MM_FMADD(a, b0, acc00);
        acc10 = MM_FMADD(a, b1, acc10);

        a = MM_BROADCAST_Sx(&ARow[4*1+2]);
        acc01 = MM_FMADD(a, b0, acc01);
        acc11 = MM_FMADD(a, b1, acc11);

        a = MM_BROADCAST_Sx(&ARow[4*2+2]);
        acc02 = MM_FMADD(a, b0, acc02);
        acc12 = MM_FMADD(a, b1, acc12);

        a = MM_BROADCAST_Sx(&ARow[4*3+2]);
        acc03 = MM_FMADD(a, b0, acc03);
        acc13 = MM_FMADD(a, b1, acc13);

        a = MM_BROADCAST_Sx(&ARow[4*4+2]);
        acc04 = MM_FMADD(a, b0, acc04);
        acc14 = MM_FMADD(a, b1, acc14);

        b0 = Bcol[0];
        b1 = Bcol[1];
        Bcol += B_WORDS_PER_ITER;

        a = MM_BROADCAST_Sx(&ARow[4*0+3]);
        acc00 = MM_FMADD(a, b0, acc00);
        acc10 = MM_FMADD(a, b1, acc10);

        a = MM_BROADCAST_Sx(&ARow[4*1+3]);
        acc01 = MM_FMADD(a, b0, acc01);
        acc11 = MM_FMADD(a, b1, acc11);

        a = MM_BROADCAST_Sx(&ARow[4*2+3]);
        acc02 = MM_FMADD(a, b0, acc02);
        acc12 = MM_FMADD(a, b1, acc12);

        a = MM_BROADCAST_Sx(&ARow[4*3+3]);
        acc03 = MM_FMADD(a, b0, acc03);
        acc13 = MM_FMADD(a, b1, acc13);

        a = MM_BROADCAST_Sx(&ARow[4*4+3]);
        acc04 = MM_FMADD(a, b0, acc04);
        acc14 = MM_FMADD(a, b1, acc14);

        ARow += 4*A_WORDS_PER_ITER;
      }

      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
      int_vector_t mask = pPrm->mask_b[lastCol];

      CCol = Crow;
      MM_STOREU_Px    (&CCol[SIMD_FACTOR*0],       MM_FMADD(acc00, alpha_ps, MM_LOADU_Px    (&CCol[SIMD_FACTOR*0])));
      MM_MASKSTOREU_Px(&CCol[SIMD_FACTOR*1], mask, MM_FMADD(acc10, alpha_ps, MM_MASKLOADU_Px(&CCol[SIMD_FACTOR*1], mask)));
      CCol += ldc;

      MM_STOREU_Px    (&CCol[SIMD_FACTOR*0],       MM_FMADD(acc01, alpha_ps, MM_LOADU_Px    (&CCol[SIMD_FACTOR*0])));
      MM_MASKSTOREU_Px(&CCol[SIMD_FACTOR*1], mask, MM_FMADD(acc11, alpha_ps, MM_MASKLOADU_Px(&CCol[SIMD_FACTOR*1], mask)));
      CCol += ldc;

      MM_STOREU_Px    (&CCol[SIMD_FACTOR*0],       MM_FMADD(acc02, alpha_ps, MM_LOADU_Px    (&CCol[SIMD_FACTOR*0])));
      MM_MASKSTOREU_Px(&CCol[SIMD_FACTOR*1], mask, MM_FMADD(acc12, alpha_ps, MM_MASKLOADU_Px(&CCol[SIMD_FACTOR*1], mask)));
      CCol += ldc;

      MM_STOREU_Px    (&CCol[SIMD_FACTOR*0],       MM_FMADD(acc03, alpha_ps, MM_LOADU_Px    (&CCol[SIMD_FACTOR*0])));
      MM_MASKSTOREU_Px(&CCol[SIMD_FACTOR*1], mask, MM_FMADD(acc13, alpha_ps, MM_MASKLOADU_Px(&CCol[SIMD_FACTOR*1], mask)));
      CCol += ldc;

      MM_STOREU_Px    (&CCol[SIMD_FACTOR*0],       MM_FMADD(acc04, alpha_ps, MM_LOADU_Px    (&CCol[SIMD_FACTOR*0])));
      MM_MASKSTOREU_Px(&CCol[SIMD_FACTOR*1], mask, MM_FMADD(acc14, alpha_ps, MM_MASKLOADU_Px(&CCol[SIMD_FACTOR*1], mask)));

      lastCol = 0;
    }
  }

  // handle remaining rows of a - non-interleaved
  for (; m < pPrm->M;  A += kSteps, C += ldc, ++m) {
    scalar_t* Crow = &C[n_bIters*B_WORDS_PER_ITER*SIMD_FACTOR];
    int lastCol = 1;
    for (int b_it = n_bIters - 1; b_it >= 0; --b_it) {
      const fp_vector_t* Bcol = &pPrm->bb[ldbb*b_it];
      const scalar_t*    ARow = reinterpret_cast<const scalar_t*>(A);
      fp_vector_t acc00 = MM_SETZERO_Px();
      fp_vector_t acc10 = MM_SETZERO_Px();
      fp_vector_t acc01 = MM_SETZERO_Px();
      fp_vector_t acc11 = MM_SETZERO_Px();
      fp_vector_t acc02 = MM_SETZERO_Px();
      fp_vector_t acc12 = MM_SETZERO_Px();
      fp_vector_t acc03 = MM_SETZERO_Px();
      fp_vector_t acc13 = MM_SETZERO_Px();

      Crow -= B_WORDS_PER_ITER*SIMD_FACTOR;
      _mm_prefetch((char*)(Crow),             _MM_HINT_T0);
      _mm_prefetch((char*)(Crow+SIMD_FACTOR), _MM_HINT_T0);

      for (int k = 0; k < kSteps; ++k) {
        fp_vector_t a;

        a = MM_BROADCAST_Sx(&ARow[0]);
        acc00 = MM_FMADD(a, Bcol[0], acc00);
        acc10 = MM_FMADD(a, Bcol[1], acc10);
        Bcol += B_WORDS_PER_ITER;

        a = MM_BROADCAST_Sx(&ARow[1]);
        acc01 = MM_FMADD(a, Bcol[0], acc01);
        acc11 = MM_FMADD(a, Bcol[1], acc11);
        Bcol += B_WORDS_PER_ITER;

        a = MM_BROADCAST_Sx(&ARow[2]);
        acc02 = MM_FMADD(a, Bcol[0], acc02);
        acc12 = MM_FMADD(a, Bcol[1], acc12);
        Bcol += B_WORDS_PER_ITER;

        a = MM_BROADCAST_Sx(&ARow[3]);
        acc03 = MM_FMADD(a, Bcol[0], acc03);
        acc13 = MM_FMADD(a, Bcol[1], acc13);
        Bcol += B_WORDS_PER_ITER;

        ARow += 4;
      }
      acc00 = MM_MADD_Px(acc00, acc01);
      acc02 = MM_MADD_Px(acc02, acc03);

      acc10 = MM_MADD_Px(acc10, acc11);
      acc12 = MM_MADD_Px(acc12, acc13);

      acc00 = MM_MADD_Px(acc00, acc02);
      acc10 = MM_MADD_Px(acc10, acc12);

      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
      int_vector_t mask = pPrm->mask_b[lastCol];

      MM_STOREU_Px    (&Crow[SIMD_FACTOR*0],       MM_FMADD(acc00, alpha_ps, MM_LOADU_Px    (&Crow[SIMD_FACTOR*0])));
      MM_MASKSTOREU_Px(&Crow[SIMD_FACTOR*1], mask, MM_FMADD(acc10, alpha_ps, MM_MASKLOADU_Px(&Crow[SIMD_FACTOR*1], mask)));

      lastCol = 0;
    }
  }
}

// minor core - inner loop processes 1 SIMD columns of B x 5 rows of A
static void fma256_noncblas_sgemm_core_mn(
 const noncblas_sgemm_prm_t* pPrm,
 scalar_t*                   C,
 int                         nRows)    // 0 < nRows <= k_step
{
  int ldc = pPrm->ldc;
  int kSteps = unsigned(nRows - 1) / 4 + 1;
  int m;
  const scalar_t* ARow = reinterpret_cast<const scalar_t*>(pPrm->aa);
  for (m = 0; m < pPrm->M - A_WORDS_PER_ITER + 1; m += A_WORDS_PER_ITER) {
    const fp_vector_t* Bcol = pPrm->bb;
    fp_vector_t acc00 = MM_SETZERO_Px();
    fp_vector_t acc10 = MM_SETZERO_Px();
    fp_vector_t acc01 = MM_SETZERO_Px();
    fp_vector_t acc11 = MM_SETZERO_Px();
    fp_vector_t acc02 = MM_SETZERO_Px();
    fp_vector_t acc12 = MM_SETZERO_Px();
    fp_vector_t acc03 = MM_SETZERO_Px();
    fp_vector_t acc13 = MM_SETZERO_Px();
    fp_vector_t acc04 = MM_SETZERO_Px();
    fp_vector_t acc14 = MM_SETZERO_Px();

    scalar_t* CCol = C;
    for (int ci = 0; ci < A_WORDS_PER_ITER; ++ci) {
      _mm_prefetch((char*)(CCol), _MM_HINT_T0);
      CCol += ldc;
    }
    for (int k = 0; k < kSteps; ++k) {
      fp_vector_t b;

      b = Bcol[0];
      acc00 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4 * 0 + 0]), b, acc00);
      acc01 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4 * 1 + 0]), b, acc01);
      acc02 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4 * 2 + 0]), b, acc02);
      acc03 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4 * 3 + 0]), b, acc03);
      acc04 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4 * 4 + 0]), b, acc04);

      b = Bcol[1];
      acc10 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4 * 0 + 1]), b, acc10);
      acc11 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4 * 1 + 1]), b, acc11);
      acc12 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4 * 2 + 1]), b, acc12);
      acc13 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4 * 3 + 1]), b, acc13);
      acc14 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4 * 4 + 1]), b, acc14);

      b = Bcol[2];
      acc00 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4 * 0 + 2]), b, acc00);
      acc01 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4 * 1 + 2]), b, acc01);
      acc02 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4 * 2 + 2]), b, acc02);
      acc03 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4 * 3 + 2]), b, acc03);
      acc04 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4 * 4 + 2]), b, acc04);

      b = Bcol[3];
      acc10 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4 * 0 + 3]), b, acc10);
      acc11 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4 * 1 + 3]), b, acc11);
      acc12 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4 * 2 + 3]), b, acc12);
      acc13 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4 * 3 + 3]), b, acc13);
      acc14 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4 * 4 + 3]), b, acc14);

      Bcol += 4;
      ARow += 4 * A_WORDS_PER_ITER;
    }
    acc00 = MM_MADD_Px(acc00, acc10);
    acc01 = MM_MADD_Px(acc01, acc11);
    acc02 = MM_MADD_Px(acc02, acc12);
    acc03 = MM_MADD_Px(acc03, acc13);
    acc04 = MM_MADD_Px(acc04, acc14);

    fp_vector_t  alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
    int_vector_t mask = pPrm->mask_b[1];

    MM_MASKSTOREU_Px(C, mask, MM_FMADD(acc00, alpha_ps, MM_MASKLOADU_Px(C, mask))); C += ldc;
    MM_MASKSTOREU_Px(C, mask, MM_FMADD(acc01, alpha_ps, MM_MASKLOADU_Px(C, mask))); C += ldc;
    MM_MASKSTOREU_Px(C, mask, MM_FMADD(acc02, alpha_ps, MM_MASKLOADU_Px(C, mask))); C += ldc;
    MM_MASKSTOREU_Px(C, mask, MM_FMADD(acc03, alpha_ps, MM_MASKLOADU_Px(C, mask))); C += ldc;
    MM_MASKSTOREU_Px(C, mask, MM_FMADD(acc04, alpha_ps, MM_MASKLOADU_Px(C, mask))); C += ldc;
  }

  // handle remaining rows of a - non-interleaved
  for (; m < pPrm->M; ++m) {
    const fp_vector_t* Bcol = pPrm->bb;
    fp_vector_t acc00 = MM_SETZERO_Px();
    fp_vector_t acc01 = MM_SETZERO_Px();
    fp_vector_t acc02 = MM_SETZERO_Px();
    fp_vector_t acc03 = MM_SETZERO_Px();

    _mm_prefetch((char*)(C), _MM_HINT_T0);
    for (int k = 0; k < kSteps; ++k) {
      acc00 = MM_FMADD(MM_BROADCAST_Sx(&ARow[0]), Bcol[0], acc00);
      acc01 = MM_FMADD(MM_BROADCAST_Sx(&ARow[1]), Bcol[1], acc01);
      acc02 = MM_FMADD(MM_BROADCAST_Sx(&ARow[2]), Bcol[2], acc02);
      acc03 = MM_FMADD(MM_BROADCAST_Sx(&ARow[3]), Bcol[3], acc03);
      Bcol += 4;
      ARow += 4;
    }
    acc00 = MM_MADD_Px(acc00, acc01);
    acc02 = MM_MADD_Px(acc02, acc03);
    acc00 = MM_MADD_Px(acc00, acc02);

    fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
    int_vector_t mask = pPrm->mask_b[1];
    MM_MASKSTOREU_Px(C, mask, MM_FMADD(acc00, alpha_ps, MM_MASKLOADU_Px(C, mask)));
    C += ldc;
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
  const scalar_t*       B, int ldb,
  int                   n_bIters,
  int                   nRows)
{
  fp_vector_t* dstCol = pPrm->bb;
  int ldbb = nRows*B_WORDS_PER_ITER;
  for (int r = 0; r < nRows; ++r) {
    const scalar_t *src = B;
    fp_vector_t* dst = dstCol;
    for (int c = 0; c < n_bIters; ++c) {
      for (int w = 0; w < B_WORDS_PER_ITER; ++w)
        dst[w] = MM_LOADU_Px(&src[w*SIMD_FACTOR]);
      src += SIMD_FACTOR*B_WORDS_PER_ITER;
      dst += ldbb;
    }
    B      += ldb;
    dstCol += B_WORDS_PER_ITER;
  }
}

static void CopyAndTransposeMjWithMask(
  noncblas_sgemm_prm_t* pPrm,
  const scalar_t *B, int ldb,
  int n_bIters, int nRows)
{
  fp_vector_t* dstCol = pPrm->bb;
  int ldbb = nRows*B_WORDS_PER_ITER;
  int_vector_t mask = pPrm->mask_b[1];
  for (int r = 0; r < nRows; ++r) {
    const scalar_t *src = B;
    fp_vector_t* dst = dstCol;
    for (int c = 0; c < n_bIters-1; ++c) {
      for (int w = 0; w < B_WORDS_PER_ITER; ++w)
        dst[w] = MM_LOADU_Px(&src[w*SIMD_FACTOR]);
      src += SIMD_FACTOR*B_WORDS_PER_ITER;
      dst += ldbb;
    }
    for (int w = 0; w < B_WORDS_PER_ITER-1; ++w)
      dst[w] = MM_LOADU_Px(&src[w*SIMD_FACTOR]);
    dst[B_WORDS_PER_ITER-1] = MM_MASKLOADU_Px(&src[(B_WORDS_PER_ITER-1)*SIMD_FACTOR], mask);
    B      += ldb;
    dstCol += B_WORDS_PER_ITER;
  }
}

static void CopyAndTransposeMnWithMask(
  noncblas_sgemm_prm_t* pPrm,
  const scalar_t *B, int ldb,
  int nRows)
{
  int_vector_t mask = pPrm->mask_b[1];
  for (int r = 0; r < nRows; B += ldb, ++r)
    pPrm->bb[r] = MM_MASKLOADU_Px(B, mask);
}

static void CopyAndTransposeA(noncblas_sgemm_prm_t* pPrm, const scalar_t *A, int n_cols)
{
  int nQuads = unsigned(n_cols-1)/4 + 1;
  int nRows = pPrm->M;
  int lda = pPrm->lda;
  int_vector4_t mask = pPrm->mask_a[(n_cols%4==0) ? 0 : 1];
  fp_vector4_t* dstRow = pPrm->aa;
  int r;
  // the bulk is interleaved in chunks of 4 scalar elements
  for (r = 0; r < nRows+1-A_WORDS_PER_ITER; dstRow += nQuads*A_WORDS_PER_ITER, r += A_WORDS_PER_ITER) {
    for (int k = 0; k < A_WORDS_PER_ITER; A += lda, ++k) {
      fp_vector4_t* dst = &dstRow[k];
      for (int wi = 0; wi < nQuads-1; dst += A_WORDS_PER_ITER, ++wi)
        *dst = MM_LOADU4_Px(&A[wi*4]);
      *dst = MM_MASKLOADU4_Px(&A[(nQuads-1)*4], mask);
    }
  }

  // remaining rows padded to multiple of 4, but not interleaved
  for (; r < nRows; dstRow += nQuads, A += lda, ++r) {
    for (int wi = 0; wi < nQuads-1; ++wi)
      dstRow[wi] = MM_LOADU4_Px(&A[wi*4]);
    dstRow[nQuads-1] = MM_MASKLOADU4_Px(&A[(nQuads-1)*4], mask);
  }
}

int gl_m_step = 0;
int gl_k_step = K_STEP;
int gl_n_superstep = N_SUPERSTEP;

// N>SIMD_FACTOR
static void noncblas_sgemm_wide_n(
  int M, int N, int K,
  scalar_t alpha,
  const scalar_t *A, int lda,
  const scalar_t *B, int ldb,
  scalar_t beta,
  scalar_t *C, int ldc)
{
  unsigned nw = unsigned(N - 1) / SIMD_FACTOR + 1;
  int nwMj = ((nw - 1) / SIMD_ELEM_PEC_COL_MJ) * SIMD_ELEM_PEC_COL_MJ;
  int nwRem = nw - nwMj;
  int nwRemMj = nwRem / B_WORDS_PER_ITER;
  int nwRemMn = nwRem - nwRemMj*B_WORDS_PER_ITER;
  int nMj = nwMj * SIMD_FACTOR;

  int k_step = gl_k_step;
  int k_Nsteps = (K*4-k_step)/(4*k_step) + 1;
  k_step = k_Nsteps < 2 ? K : ((K-1)/(k_Nsteps*4) + 1) * 4;

  int m_step = gl_m_step;
  if (m_step == 0) // auto m_step
    m_step = MxN_BLOCK_SZ/(N*sizeof(scalar_t));

  m_step = std::max(m_step, 20);
  int m_Nsteps = m_step > 0 ? (M*2-m_step)/(2*m_step) + 1 : 1;
  m_step = m_Nsteps < 2 ? M : ((M-1)/(m_Nsteps*A_WORDS_PER_ITER) + 1) * A_WORDS_PER_ITER;

  int n_superstep = N;
  int n_superstepReq = gl_n_superstep;
  if (n_superstepReq > n_step && n_superstepReq < N) {
    int n_Nsupersteps = (N*2-n_superstepReq)/(2*n_superstepReq) + 1;
    if (n_Nsupersteps >= 2) {
      int n_Nsteps = (N-1)/n_step + 1;
      int n_stepsPerSuper = n_Nsteps/n_Nsupersteps;
      if ((n_Nsteps-1) % n_stepsPerSuper <= (n_Nsteps-1) % (n_stepsPerSuper+1))
        n_stepsPerSuper += 1;
      n_superstep = n_stepsPerSuper * n_step;
    }
  }

  const int bb_sz = SIMD_ELEM_PEC_COL_MJ*k_step;
  const int aa_sz = (m_step*k_step-1)/SIMD_FACTOR + 1;
  char* aabb = new char[(bb_sz+aa_sz+1)*sizeof(fp_vector_t)];
  uintptr_t adj = (0-uintptr_t(aabb)) % sizeof(fp_vector_t);

  noncblas_sgemm_prm_t prm;
  prm.bb = reinterpret_cast<fp_vector_t*>(aabb+adj);
  prm.aa = reinterpret_cast<fp_vector4_t*>(&prm.bb[bb_sz]);
  prm.lda = lda;
  prm.ldc = ldc;
  prm.alpha = alpha;
  memset(&prm.mask_b[0], -1, sizeof(prm.mask_b[0]));
  prm.mask_b[2] = prm.mask_b[0];
  unsigned remW_n = nw*SIMD_FACTOR - N;
  if (remW_n > 0) // mask off elements of rightmost SIMD word in B and C
    memset((char*)&prm.mask_b[3] - sizeof(*C)*remW_n, 0, sizeof(*C)*remW_n);

  memset(&prm.mask_a[0], -1, sizeof(prm.mask_a[0]));
  prm.mask_a[1] = prm.mask_a[0];
  unsigned remW_k = (4 - unsigned(K)) % 4;
  if (remW_k > 0) // mask off elements of rightmost fp_vector4_t word in A
    memset((char*)&prm.mask_a[2] - sizeof(*A)*remW_k, 0, sizeof(*A)*remW_k);

  // printf("nMaj=%d, nwRemMn=%d, nwRemMj=%d remW_n=%d n_step=%d n_superstep=%d\n", nMj, nwRemMn, nwRemMj, remW_n, n_step, n_superstep);

  for (int nBeg = 0; nBeg < N; nBeg += n_superstep) {
    int delta_n = std::min(n_superstep, N-nBeg);
    for (int k = 0; k < K; k += k_step) {
      int delta_k = std::min(int(k_step), K - k);
      for (int m = 0; m < M; m += prm.M) {
        prm.M = std::min(m_step, M - m);

        scalar_t *Crow = &C[m*ldc];
        if (k==0)
          fma256_noncblas_sgemm_multC(prm.M, delta_n, beta, &Crow[nBeg], ldc);

        prm.mask_b[1] = prm.mask_b[0]; // all words in use
        CopyAndTransposeA(&prm, &A[m*lda+k], delta_k);

        int nEnd = std::min(nBeg + delta_n, nMj);
        int n;
        for (n = nBeg; n < nEnd; n += n_step) {
          // process full-width major rectangles
          CopyAndTransposeMj(&prm, &B[k*ldb + n], ldb, N_STEP_MULTIPLIER, delta_k);
          fma256_noncblas_sgemm_core_mj(&prm, &Crow[n], N_STEP_MULTIPLIER, delta_k);
        }
        if (n >= nMj) {
          if (nwRemMn > 0) {
            if (nwRemMj == 0)
              prm.mask_b[1] = prm.mask_b[2]; // mask for leftmost word
            CopyAndTransposeMnWithMask(&prm, &B[k*ldb + n], ldb, delta_k);
            fma256_noncblas_sgemm_core_mn(&prm, &Crow[n], delta_k);
            n += SIMD_FACTOR;
          }
          if (nwRemMj > 0) {
            prm.mask_b[1] = prm.mask_b[2]; // mask for leftmost word
            CopyAndTransposeMjWithMask(&prm, &B[k*ldb + n], ldb, nwRemMj, delta_k);
            fma256_noncblas_sgemm_core_mj(&prm, &Crow[n], nwRemMj, delta_k);
          }
        }
      }
    }
  }
  delete [] aabb;
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
  if (N > SIMD_FACTOR) {
    noncblas_sgemm_wide_n(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  } else if (N >= 1) {
    noncblas_sgemm_narrow_n(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }
}
