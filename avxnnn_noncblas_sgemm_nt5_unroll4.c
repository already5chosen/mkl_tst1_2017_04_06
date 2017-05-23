enum {
 SIMD_FACTOR          = sizeof(fp_vector_t)/sizeof(scalar_t),
 A_WORDS_PER_ITER     = 5,
 B_WORDS_PER_ITER     = 2,
 SIMD_ELEM_PEC_COL_MJ = B_WORDS_PER_ITER*N_STEP_MULTIPLIER,
 n_step               = SIMD_ELEM_PEC_COL_MJ*SIMD_FACTOR,
 N_SUPERSTEP          = n_step*N_SUPER_STEP_MULTIPLIER,
};

enum {
  C_OPTION_UPDATE, C_OPTION_REPLACE, C_OPTION_MULTIPLY
};

typedef struct {
  int           M;
  int           lda;
  int           ldc;
  int           ldcc;
  int           c_option;
  int           masked_n_w;
  scalar_t      alpha;
  scalar_t      beta;
  int_vector_t  mask_n;
  fp_vector_t*  bb;        // [B_WORDS_PER_ITER*k_step*N_SUPER_STEP_MULTIPLIER];
  scalar_t*     aa;        // [k_step*m_step_max];
  fp_vector_t*  cc;        // [(SIMD_ELEM_PEC_COL_MJ*N_SUPER_STEP_MULTIPLIER)*m_step_max];
} noncblas_sgemm_prm_t;

// major core - inner loop processes 2 SIMD columns of B x 5 rows of A
static void fma256_noncblas_sgemm_core_mj(
 const noncblas_sgemm_prm_t* pPrm,
 int                         n_bIters, // 0 < n_bIters <= N_STEP_MULTIPLIER
 int                         nRows)    // 0 < nRows    <= k_step
{
  const int ldcc = pPrm->ldcc;
  int kSteps = (unsigned)(nRows-1) / 4;
  int kRem   = (unsigned)(nRows-1) % 4;
  int ldbb   = B_WORDS_PER_ITER*nRows;
  int m;
  const scalar_t* A = pPrm->aa;
  fp_vector_t* C = pPrm->cc;
  for (m = 0; m < pPrm->M-A_WORDS_PER_ITER+1;
    A += nRows*A_WORDS_PER_ITER,
    C += ldcc*A_WORDS_PER_ITER,
    m += A_WORDS_PER_ITER) {
    fp_vector_t* Crow = C;
#if 0
    #define Prefetch2words(x) \
      _mm_prefetch((char*)(x) + 0,                       _MM_HINT_T0); \
      _mm_prefetch((char*)(x) + sizeof(fp_vector_t)*2-1, _MM_HINT_T0);
#else
    #define Prefetch2words(x)
#endif
    for (int b_it = 0; b_it < n_bIters; Crow += B_WORDS_PER_ITER, ++b_it) {
      const fp_vector_t* Bcol = &pPrm->bb[ldbb*b_it];
      const scalar_t* ARow = A;
      fp_vector_t* CPrefetch = Crow;

      fp_vector_t a;
      fp_vector_t b0 = Bcol[0];
      fp_vector_t b1 = Bcol[1];
      Bcol += B_WORDS_PER_ITER;

      a = MM_BROADCAST_Sx(&ARow[0]);
      fp_vector_t acc00 = MM_MUL_Px(a, b0);
      fp_vector_t acc10 = MM_MUL_Px(a, b1);
      Prefetch2words(CPrefetch); CPrefetch += ldcc;

      a = MM_BROADCAST_Sx(&ARow[1]);
      fp_vector_t acc01 = MM_MUL_Px(a, b0);
      fp_vector_t acc11 = MM_MUL_Px(a, b1);
      Prefetch2words(CPrefetch); CPrefetch += ldcc;

      a = MM_BROADCAST_Sx(&ARow[2]);
      fp_vector_t acc02 = MM_MUL_Px(a, b0);
      fp_vector_t acc12 = MM_MUL_Px(a, b1);
      Prefetch2words(CPrefetch); CPrefetch += ldcc;

      a = MM_BROADCAST_Sx(&ARow[3]);
      fp_vector_t acc03 = MM_MUL_Px(a, b0);
      fp_vector_t acc13 = MM_MUL_Px(a, b1);
      Prefetch2words(CPrefetch); CPrefetch += ldcc;

      a = MM_BROADCAST_Sx(&ARow[4]);
      fp_vector_t acc04 = MM_MUL_Px(a, b0);
      fp_vector_t acc14 = MM_MUL_Px(a, b1);
      Prefetch2words(CPrefetch); CPrefetch += ldcc;

      ARow += A_WORDS_PER_ITER;

      int k = kSteps;
      do {
        b0 = Bcol[0];
        b1 = Bcol[1];
        Bcol += B_WORDS_PER_ITER;

        a = MM_BROADCAST_Sx(&ARow[5*0+0]);
        acc00 = MM_FMADD(a, b0, acc00);
        acc10 = MM_FMADD(a, b1, acc10);

        a = MM_BROADCAST_Sx(&ARow[5*0+1]);
        acc01 = MM_FMADD(a, b0, acc01);
        acc11 = MM_FMADD(a, b1, acc11);

        a = MM_BROADCAST_Sx(&ARow[5*0+2]);
        acc02 = MM_FMADD(a, b0, acc02);
        acc12 = MM_FMADD(a, b1, acc12);

        a = MM_BROADCAST_Sx(&ARow[5*0+3]);
        acc03 = MM_FMADD(a, b0, acc03);
        acc13 = MM_FMADD(a, b1, acc13);

        a = MM_BROADCAST_Sx(&ARow[5*0+4]);
        acc04 = MM_FMADD(a, b0, acc04);
        acc14 = MM_FMADD(a, b1, acc14);

        b0 = Bcol[0];
        b1 = Bcol[1];
        Bcol += B_WORDS_PER_ITER;

        a = MM_BROADCAST_Sx(&ARow[5*1+0]);
        acc00 = MM_FMADD(a, b0, acc00);
        acc10 = MM_FMADD(a, b1, acc10);

        a = MM_BROADCAST_Sx(&ARow[5*1+1]);
        acc01 = MM_FMADD(a, b0, acc01);
        acc11 = MM_FMADD(a, b1, acc11);

        a = MM_BROADCAST_Sx(&ARow[5*1+2]);
        acc02 = MM_FMADD(a, b0, acc02);
        acc12 = MM_FMADD(a, b1, acc12);

        a = MM_BROADCAST_Sx(&ARow[5*1+3]);
        acc03 = MM_FMADD(a, b0, acc03);
        acc13 = MM_FMADD(a, b1, acc13);

        a = MM_BROADCAST_Sx(&ARow[5*1+4]);
        acc04 = MM_FMADD(a, b0, acc04);
        acc14 = MM_FMADD(a, b1, acc14);

        b0 = Bcol[0];
        b1 = Bcol[1];
        Bcol += B_WORDS_PER_ITER;

        a = MM_BROADCAST_Sx(&ARow[5*2+0]);
        acc00 = MM_FMADD(a, b0, acc00);
        acc10 = MM_FMADD(a, b1, acc10);

        a = MM_BROADCAST_Sx(&ARow[5*2+1]);
        acc01 = MM_FMADD(a, b0, acc01);
        acc11 = MM_FMADD(a, b1, acc11);

        a = MM_BROADCAST_Sx(&ARow[5*2+2]);
        acc02 = MM_FMADD(a, b0, acc02);
        acc12 = MM_FMADD(a, b1, acc12);

        a = MM_BROADCAST_Sx(&ARow[5*2+3]);
        acc03 = MM_FMADD(a, b0, acc03);
        acc13 = MM_FMADD(a, b1, acc13);

        a = MM_BROADCAST_Sx(&ARow[5*2+4]);
        acc04 = MM_FMADD(a, b0, acc04);
        acc14 = MM_FMADD(a, b1, acc14);

        b0 = Bcol[0];
        b1 = Bcol[1];
        Bcol += B_WORDS_PER_ITER;

        a = MM_BROADCAST_Sx(&ARow[5*3+0]);
        acc00 = MM_FMADD(a, b0, acc00);
        acc10 = MM_FMADD(a, b1, acc10);

        a = MM_BROADCAST_Sx(&ARow[5*3+1]);
        acc01 = MM_FMADD(a, b0, acc01);
        acc11 = MM_FMADD(a, b1, acc11);

        a = MM_BROADCAST_Sx(&ARow[5*3+2]);
        acc02 = MM_FMADD(a, b0, acc02);
        acc12 = MM_FMADD(a, b1, acc12);

        a = MM_BROADCAST_Sx(&ARow[5*3+3]);
        acc03 = MM_FMADD(a, b0, acc03);
        acc13 = MM_FMADD(a, b1, acc13);

        a = MM_BROADCAST_Sx(&ARow[5*3+4]);
        acc04 = MM_FMADD(a, b0, acc04);
        acc14 = MM_FMADD(a, b1, acc14);

        ARow += 4*A_WORDS_PER_ITER;
      } while (--k);

      if (kRem != 0) {
        fp_vector_t a, b0, b1;
        b0 = Bcol[0];
        b1 = Bcol[1];
        Bcol += B_WORDS_PER_ITER;

        a = MM_BROADCAST_Sx(&ARow[5*0+0]);
        acc00 = MM_FMADD(a, b0, acc00);
        acc10 = MM_FMADD(a, b1, acc10);

        a = MM_BROADCAST_Sx(&ARow[5*0+1]);
        acc01 = MM_FMADD(a, b0, acc01);
        acc11 = MM_FMADD(a, b1, acc11);

        a = MM_BROADCAST_Sx(&ARow[5*0+2]);
        acc02 = MM_FMADD(a, b0, acc02);
        acc12 = MM_FMADD(a, b1, acc12);

        a = MM_BROADCAST_Sx(&ARow[5*0+3]);
        acc03 = MM_FMADD(a, b0, acc03);
        acc13 = MM_FMADD(a, b1, acc13);

        a = MM_BROADCAST_Sx(&ARow[5*0+4]);
        acc04 = MM_FMADD(a, b0, acc04);
        acc14 = MM_FMADD(a, b1, acc14);

        if (kRem != 1) {
          b0 = Bcol[0];
          b1 = Bcol[1];
          Bcol += B_WORDS_PER_ITER;

          a = MM_BROADCAST_Sx(&ARow[5*1+0]);
          acc00 = MM_FMADD(a, b0, acc00);
          acc10 = MM_FMADD(a, b1, acc10);

          a = MM_BROADCAST_Sx(&ARow[5*1+1]);
          acc01 = MM_FMADD(a, b0, acc01);
          acc11 = MM_FMADD(a, b1, acc11);

          a = MM_BROADCAST_Sx(&ARow[5*1+2]);
          acc02 = MM_FMADD(a, b0, acc02);
          acc12 = MM_FMADD(a, b1, acc12);

          a = MM_BROADCAST_Sx(&ARow[5*1+3]);
          acc03 = MM_FMADD(a, b0, acc03);
          acc13 = MM_FMADD(a, b1, acc13);

          a = MM_BROADCAST_Sx(&ARow[5*1+4]);
          acc04 = MM_FMADD(a, b0, acc04);
          acc14 = MM_FMADD(a, b1, acc14);

          if (kRem != 2) {
            b0 = Bcol[0];
            b1 = Bcol[1];
            Bcol += B_WORDS_PER_ITER;

            a = MM_BROADCAST_Sx(&ARow[5*2+0]);
            acc00 = MM_FMADD(a, b0, acc00);
            acc10 = MM_FMADD(a, b1, acc10);

            a = MM_BROADCAST_Sx(&ARow[5*2+1]);
            acc01 = MM_FMADD(a, b0, acc01);
            acc11 = MM_FMADD(a, b1, acc11);

            a = MM_BROADCAST_Sx(&ARow[5*2+2]);
            acc02 = MM_FMADD(a, b0, acc02);
            acc12 = MM_FMADD(a, b1, acc12);

            a = MM_BROADCAST_Sx(&ARow[5*2+3]);
            acc03 = MM_FMADD(a, b0, acc03);
            acc13 = MM_FMADD(a, b1, acc13);

            a = MM_BROADCAST_Sx(&ARow[5*2+4]);
            acc04 = MM_FMADD(a, b0, acc04);
            acc14 = MM_FMADD(a, b1, acc14);
          }
        }
      }

      fp_vector_t* CCol = Crow;
      CCol[0] = acc00; CCol[1] = acc10; CCol += ldcc;
      CCol[0] = acc01; CCol[1] = acc11; CCol += ldcc;
      CCol[0] = acc02; CCol[1] = acc12; CCol += ldcc;
      CCol[0] = acc03; CCol[1] = acc13; CCol += ldcc;
      CCol[0] = acc04; CCol[1] = acc14;
    }
  }

  // handle remaining rows of a - non-interleaved
  kSteps = (unsigned)(nRows) / 4;
  kRem   = (unsigned)(nRows) % 4;
  for (; m < pPrm->M;  A += nRows, C += ldcc, ++m) {
    fp_vector_t* Crow = C;
    for (int b_it = 0; b_it < n_bIters; Crow += B_WORDS_PER_ITER, ++b_it) {
      const fp_vector_t* Bcol = &pPrm->bb[ldbb*b_it];
      const scalar_t*    ARow = (const scalar_t*)(A);
      fp_vector_t acc00 = MM_SETZERO_Px();
      fp_vector_t acc10 = MM_SETZERO_Px();
      fp_vector_t acc01 = MM_SETZERO_Px();
      fp_vector_t acc11 = MM_SETZERO_Px();
      fp_vector_t acc02 = MM_SETZERO_Px();
      fp_vector_t acc12 = MM_SETZERO_Px();
      fp_vector_t acc03 = MM_SETZERO_Px();
      fp_vector_t acc13 = MM_SETZERO_Px();

      Prefetch2words(Crow);

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
      if (kRem != 0) {
        fp_vector_t a;
        a = MM_BROADCAST_Sx(&ARow[0]);
        acc00 = MM_FMADD(a, Bcol[0], acc00);
        acc10 = MM_FMADD(a, Bcol[1], acc10);
        Bcol += B_WORDS_PER_ITER;
        if (kRem != 1) {
          a = MM_BROADCAST_Sx(&ARow[1]);
          acc01 = MM_FMADD(a, Bcol[0], acc01);
          acc11 = MM_FMADD(a, Bcol[1], acc11);
          Bcol += B_WORDS_PER_ITER;
          if (kRem != 2) {
            a = MM_BROADCAST_Sx(&ARow[2]);
            acc02 = MM_FMADD(a, Bcol[0], acc02);
            acc12 = MM_FMADD(a, Bcol[1], acc12);
            Bcol += B_WORDS_PER_ITER;
          }
        }
      }
      acc00 = MM_ADD_Px(acc00, acc01);
      acc02 = MM_ADD_Px(acc02, acc03);

      acc10 = MM_ADD_Px(acc10, acc11);
      acc12 = MM_ADD_Px(acc12, acc13);

      acc00 = MM_ADD_Px(acc00, acc02);
      acc10 = MM_ADD_Px(acc10, acc12);

      Crow[0] = acc00;
      Crow[1] = acc10;
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
      // 'gcc -O1' does not generate good code for memcpy
      // On the other hand, MSVC does not generate good code for loop
      // Since these couple of lines is performance-critical and can easily cost 6-7% in time
      // I coded it in ugly manner with ifdef, to please each compiler with its preferred construct
      #ifdef _MSC_VER
      memcpy(dst, src, sizeof(*dst)*B_WORDS_PER_ITER);
      #else
      for (int w = 0; w < B_WORDS_PER_ITER; ++w)
        dst[w] = MM_LOADU_Px(&src[w*SIMD_FACTOR]);
      #endif
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
  int   n_bIters,
  int   nRows)
{
  fp_vector_t* dstCol = pPrm->bb;
  int ldbb = nRows*B_WORDS_PER_ITER;
  int_vector_t mask = pPrm->mask_n;
  int masked_n_w = pPrm->masked_n_w;
  int n_bFullIters = n_bIters - (masked_n_w >= 0);
  for (int r = 0; r < nRows; ++r) {
    const scalar_t *src = B;
    fp_vector_t* dst = dstCol;
    for (int c = 0; c < n_bFullIters; ++c) {
      for (int w = 0; w < B_WORDS_PER_ITER; ++w)
        dst[w] = MM_LOADU_Px(&src[w*SIMD_FACTOR]);
      src += SIMD_FACTOR*B_WORDS_PER_ITER;
      dst += ldbb;
    }
    if (masked_n_w >= 0) {
      for (int w = 0; w < masked_n_w; ++w)
        dst[w] = MM_LOADU_Px(&src[w*SIMD_FACTOR]);
      fp_vector_t lastB = MM_MASKLOADU_Px(&src[masked_n_w*SIMD_FACTOR], mask);
      for (int w = masked_n_w; w < B_WORDS_PER_ITER; ++w)
        dst[masked_n_w] = lastB;
    }
    B      += ldb;
    dstCol += B_WORDS_PER_ITER;
  }
}

#if 0
static void CopyAndTransposeMnWithMask(
  noncblas_sgemm_prm_t* pPrm,
  const scalar_t *B, int ldb,
  int nRows)
{
  int_vector_t mask = pPrm->mask_b[1];
  for (int r = 0; r < nRows; B += ldb, ++r)
    pPrm->bb[r] = MM_MASKLOADU_Px(B, mask);
}
#endif

static void CopyAndInterleaveA(noncblas_sgemm_prm_t* pPrm, const scalar_t *A, int n_cols)
{
  int lda1 = pPrm->lda;
  int lda2 = lda1 + lda1;
  int lda3 = lda2 + lda1;
  int lda4 = lda3 + lda1;
  int lda5 = lda4 + lda1;
  scalar_t* dst = pPrm->aa;
  int r;
  // the bulk is interleaved
  for (r = pPrm->M+1-A_WORDS_PER_ITER; r > 0; A += lda5, r -= A_WORDS_PER_ITER) {
    const scalar_t *src = A;
    for (int k = 0; k < n_cols; src += 1, dst += A_WORDS_PER_ITER, ++k) {
      scalar_t a0 = src[0];
      scalar_t a1 = src[lda1];
      scalar_t a2 = src[lda2];
      scalar_t a3 = src[lda3];
      scalar_t a4 = src[lda4];
      dst[0] = a0;
      dst[1] = a1;
      dst[2] = a2;
      dst[3] = a3;
      dst[4] = a4;
    }
  }

  // remaining rows not interleaved
  r += A_WORDS_PER_ITER - 1;
  for (; r > 0; A += lda1, dst += n_cols, --r) {
    memcpy(dst, A, sizeof(scalar_t)*n_cols);
  }
}

#if 1
static void UpdateC(noncblas_sgemm_prm_t* pPrm, scalar_t *C, int n_cols)
{
  int m = pPrm->M;
  const fp_vector_t* cc = pPrm->cc;
  int_vector_t mask = pPrm->mask_n;
  fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
  const int ldc = pPrm->ldc;
  do {
    int n = (unsigned)n_cols / SIMD_FACTOR;
    const fp_vector_t* src = cc;
    scalar_t*          dst = C;
    do {
      MM_STOREU_Px(dst, MM_FMADD(*src, alpha_ps, MM_LOADU_Px(dst)));
      //_mm_prefetch((char*)(dst+ldc), _MM_HINT_T0);
      dst += SIMD_FACTOR;
      src += 1;
    } while (--n);
    if ((unsigned)n_cols % SIMD_FACTOR)
      MM_MASKSTOREU_Px(dst, mask, MM_FMADD(*src, alpha_ps, MM_MASKLOADU_Px(dst, mask)));

    cc += pPrm->ldcc;
    C  += ldc;
  } while (--m);
}
#else
static void UpdateC(noncblas_sgemm_prm_t* pPrm, scalar_t *C, int n_cols)
{
  int m = pPrm->M;
  const fp_vector_t* cc = pPrm->cc;
  int_vector_t mask = pPrm->mask_n;
  fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
  const int ldc = pPrm->ldc;
  unsigned nw = (unsigned)n_cols / SIMD_FACTOR;
  unsigned nwh = nw / 2;
  unsigned nwr = nw % 2;
  unsigned nr = (unsigned)n_cols % SIMD_FACTOR;
  do {
    const fp_vector_t* src = cc;
    scalar_t*          dst = C;
    int n  = nwh;
    do {
      fp_vector_t c0 = MM_LOADU_Px(dst+SIMD_FACTOR*0);
      fp_vector_t c1 = MM_LOADU_Px(dst+SIMD_FACTOR*1);
      MM_STOREU_Px(dst+SIMD_FACTOR*0, MM_FMADD(src[0], alpha_ps, c0));
      MM_STOREU_Px(dst+SIMD_FACTOR*1, MM_FMADD(src[1], alpha_ps, c1));
      dst += SIMD_FACTOR*2;
      src += 2;
    } while (--n);
    if (nwr) {
      MM_STOREU_Px(dst, MM_FMADD(*src, alpha_ps, MM_LOADU_Px(dst)));
      dst += SIMD_FACTOR;
      src += 1;
    }
    if (nr)
      MM_MASKSTOREU_Px(dst, mask, MM_FMADD(*src, alpha_ps, MM_MASKLOADU_Px(dst, mask)));

    cc += pPrm->ldcc;
    C  += ldc;
  } while (--m);
}
#endif

static void ReplaceC(noncblas_sgemm_prm_t* pPrm, scalar_t *C, int n_cols)
{
  int m = pPrm->M;
  const fp_vector_t* cc = pPrm->cc;
  int_vector_t mask = pPrm->mask_n;
  fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
  const int ldc = pPrm->ldc;
  do {
    int n = (unsigned)n_cols / SIMD_FACTOR;
    const fp_vector_t* src = cc;
    scalar_t*          dst = C;
    do {
      MM_STOREU_Px(dst, MM_MUL_Px(*src, alpha_ps));
      //_mm_prefetch((char*)(dst+ldc), _MM_HINT_T0);
      dst += SIMD_FACTOR;
      src += 1;
    } while (--n);
    if ((unsigned)n_cols % SIMD_FACTOR)
      MM_MASKSTOREU_Px(dst, mask, MM_MUL_Px(*src, alpha_ps));

    cc += pPrm->ldcc;
    C  += ldc;
  } while (--m);
}

static void MultiplyC(noncblas_sgemm_prm_t* pPrm, scalar_t *C, int n_cols)
{
  int m = pPrm->M;
  const fp_vector_t* cc = pPrm->cc;
  int_vector_t mask = pPrm->mask_n;
  fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
  fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
  const int ldc = pPrm->ldc;
  do {
    int n = (unsigned)n_cols / SIMD_FACTOR;
    const fp_vector_t* src = cc;
    scalar_t*          dst = C;
    do {
      MM_STOREU_Px(dst, MM_FMADD(*src, alpha_ps, MM_MUL_Px(MM_LOADU_Px(dst), beta_ps)));
      //_mm_prefetch((char*)(dst+ldc), _MM_HINT_T0);
      dst += SIMD_FACTOR;
      src += 1;
    } while (--n);
    if ((unsigned)n_cols % SIMD_FACTOR)
      MM_MASKSTOREU_Px(dst, mask, MM_FMADD(*src, alpha_ps, MM_MUL_Px(MM_MASKLOADU_Px(dst, mask), beta_ps)));

    cc += pPrm->ldcc;
    C  += ldc;
  } while (--m);
}


static int st_m_step = 0;
static int st_k_step = 0;
extern uint64_t dbg_tt;
#include <stdio.h>
#include <x86intrin.h>

// N>SIMD_FACTOR
static void noncblas_sgemm_wide_n(
  int M, int N, int K,
  scalar_t alpha,
  const scalar_t *A, int lda,
  const scalar_t *B, int ldb,
  scalar_t beta,
  scalar_t *C, int ldc)
{
  const int K_STEP_NOM = (K_STEP/4)*4 + 1;
  const int K_STEP_MAX = (K_STEP_NOM/2)*3;
  int k_step = K > K_STEP_NOM ? K_STEP_NOM : K;

  const int M_STEP_NOM = ((M_STEP-1)/A_WORDS_PER_ITER + 1) * A_WORDS_PER_ITER;
  const int M_STEP_MAX = (M_STEP_NOM/2)*3;
  int m_step_nom = M > M_STEP_NOM ? M_STEP_NOM : M;

  const int N_SUPERSTEP_NOM = N_SUPERSTEP;
  const int N_SUPERSTEP_MAX = (N_SUPERSTEP_NOM/2)*3;
  int n_superstep = N > N_SUPERSTEP_NOM ? N_SUPERSTEP_NOM : N;

  const int n_super_step_multiplier = (n_superstep-1)/n_step+1;
  const int aa_sz = (m_step_nom*k_step-1)/SIMD_FACTOR + 1;
  const int bb_sz = SIMD_ELEM_PEC_COL_MJ*n_super_step_multiplier*k_step;
  const int cc_sz = SIMD_ELEM_PEC_COL_MJ*n_super_step_multiplier*m_step_nom;
  const int workBufSz = aa_sz + bb_sz + cc_sz;
  // I didn't find a standard portable way to allocate 32-byte aligned buffer
  // So I am doing it in hackish, but reliable way
  char* workBufAlloc = malloc((workBufSz+1)*sizeof(fp_vector_t));
  uintptr_t workBufAdj = (0-(uintptr_t)(workBufAlloc)) % sizeof(fp_vector_t);
  fp_vector_t* workBuf = (fp_vector_t*)(workBufAlloc+workBufAdj);

  noncblas_sgemm_prm_t prm;
  prm.aa = (scalar_t*)    (workBuf+0);
  fp_vector_t* bb_beg = workBuf+aa_sz;
  prm.lda   = lda;
  prm.ldc   = ldc;
  prm.alpha = alpha;
  prm.beta  = beta;

  memset(&prm.mask_n, -1, sizeof(prm.mask_n));
  prm.masked_n_w = -1;
  unsigned nRem = (unsigned)N % (SIMD_FACTOR*B_WORDS_PER_ITER);
  if (nRem > 0) {
    prm.masked_n_w = nRem / SIMD_FACTOR;
    nRem %= SIMD_FACTOR;
    if (nRem > 0) {
      // mask off elements of rightmost SIMD word in B and C
      memset(&prm.mask_n, 0, sizeof(prm.mask_n));
      memset((char*)&prm.mask_n, -1, sizeof(*C)*nRem);
    }
  }

  //printf("nMj=%d, nwRemMn=%d, nwRemMj=%d remW_n=%d n_step=%d\n", nMj, nwRemMn, nwRemMj, remW_n, n_step);
  uint64_t tt = 0;

  for (int n = 0; n < N; n += n_superstep) {
    int delta_n = N - n;
    if (delta_n > n_superstep) {
      if (delta_n < N_SUPERSTEP_MAX)
        n_superstep = ((unsigned)delta_n/(n_step*2)+1)*n_step;
      delta_n = n_superstep;
    }

    int n_steps = (unsigned)(delta_n-1)/n_step + 1;
    prm.ldcc = n_steps*SIMD_ELEM_PEC_COL_MJ;

    for (int k = 0; k < K; k += k_step) {
      prm.c_option = C_OPTION_UPDATE;
      if (k==0 && prm.beta != 1.0f)
        prm.c_option = (prm.beta == 0) ? C_OPTION_REPLACE : C_OPTION_MULTIPLY;
      int delta_k = K - k;
      if (delta_k > k_step) {
        if (delta_k < K_STEP_MAX)
          k_step = ((unsigned)(delta_k)/8+1) * 4 + 1;
        delta_k = k_step;
      }

      fp_vector_t* cc_beg = &bb_beg[SIMD_ELEM_PEC_COL_MJ*k_step*n_steps];
      int m_step = m_step_nom;
      for (int m = 0; m < M; m += prm.M) {
        int delta_m = M - m;
        if (delta_m > m_step) {
          if (delta_m < M_STEP_MAX)
            m_step = ((unsigned)delta_m/(A_WORDS_PER_ITER*2)+1)*A_WORDS_PER_ITER;
          delta_m = m_step;
        }
        prm.M = delta_m;

        CopyAndInterleaveA(&prm, &A[m*lda+k], delta_k);

        const scalar_t *BRow = &B[k*ldb+n];
        prm.bb = bb_beg;
        prm.cc = cc_beg;
        for (unsigned ni = (unsigned)delta_n / n_step;
          ni > 0;
          --ni,
          BRow   += n_step,
          prm.bb += SIMD_ELEM_PEC_COL_MJ*k_step,
          prm.cc += SIMD_ELEM_PEC_COL_MJ) {
          if (m == 0)
            CopyAndTransposeMj(&prm, BRow, ldb, N_STEP_MULTIPLIER, delta_k);
          // uint64_t t0 = __rdtsc();
          fma256_noncblas_sgemm_core_mj(&prm, N_STEP_MULTIPLIER, delta_k);
          // uint64_t t1 = __rdtsc();
          // tt += t1 - t0;
        }

        unsigned nStepRem = (unsigned)delta_n % n_step;
        if (nStepRem != 0) {
          int n_bIters = (nStepRem - 1) / (B_WORDS_PER_ITER*SIMD_FACTOR) + 1;
          if (m == 0)
            CopyAndTransposeMjWithMask(&prm, BRow, ldb, n_bIters, delta_k);
          // uint64_t t0 = __rdtsc();
          fma256_noncblas_sgemm_core_mj(&prm, n_bIters, delta_k);
          // uint64_t t1 = __rdtsc();
          // tt += t1 - t0;
        }

        uint64_t t0 = __rdtsc();
        prm.cc = cc_beg;
        scalar_t *CRow = &C[m*ldc + n];
        if (prm.c_option == C_OPTION_UPDATE)
          UpdateC(&prm, CRow, delta_n);
        else if (prm.c_option == C_OPTION_REPLACE)
          ReplaceC(&prm, CRow, delta_n);
        else
          MultiplyC(&prm, CRow, delta_n);
        uint64_t t1 = __rdtsc();
        tt += t1 - t0;
      }
    }
  }
  free(workBufAlloc);
  dbg_tt = tt;
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

void tune_name(int m_step, int k_step) {
  st_m_step = m_step;
  st_k_step = k_step;
}