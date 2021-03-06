enum {
 SIMD_FACTOR          = sizeof(fp_vector_t)/sizeof(scalar_t),
};


typedef struct {
  const scalar_t *A;
  const scalar_t *B;
  scalar_t *C;
  scalar_t alpha;
  scalar_t beta;
  int M, N, K, lda, ldb, ldc;
  int_vector_t mask_n[1];
} gemm_prm_t;


static void saxpy_5x2(const gemm_prm_t* pPrm, const scalar_t *Arow, const scalar_t *Brow, scalar_t *Cr)
{
  unsigned N = pPrm->N;
  unsigned nLoopCnt = N / SIMD_FACTOR;
  unsigned nRem     = N % SIMD_FACTOR;

  fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
  fp_vector_t a00 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[0]), alpha_ps);
  fp_vector_t a01 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[1]), alpha_ps);
  fp_vector_t a02 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[2]), alpha_ps);
  fp_vector_t a03 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[3]), alpha_ps);
  fp_vector_t a04 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[4]), alpha_ps);
  Arow += pPrm->lda;
  fp_vector_t a10 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[0]), alpha_ps);
  fp_vector_t a11 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[1]), alpha_ps);
  fp_vector_t a12 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[2]), alpha_ps);
  fp_vector_t a13 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[3]), alpha_ps);
  fp_vector_t a14 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[4]), alpha_ps);
  Arow += pPrm->lda;
  
  const int ldb1 = pPrm->ldb;
  const int ldb2 = ldb1+ldb1;
  const int ldb3 = ldb2+ldb1;
  const int ldb4 = ldb3+ldb1;
  // const int ldb5 = ldb4+ldb1;
  const int ldc  = pPrm->ldc;
  
  if (nLoopCnt) {
    unsigned n = nLoopCnt;
    do {
      fp_vector_t c0 = MM_LOADU_Px(&Cr[0]);
      fp_vector_t c1 = MM_LOADU_Px(&Cr[ldc]);
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

      b = MM_LOADU_Px(&Brow[ldb3]);
      c0 = MM_FMADD(b, a03, c0);
      c1 = MM_FMADD(b, a13, c1);

      b = MM_LOADU_Px(&Brow[ldb4]);
      c0 = MM_FMADD(b, a04, c0);
      c1 = MM_FMADD(b, a14, c1);

      MM_STOREU_Px(&Cr[0],   c0);
      MM_STOREU_Px(&Cr[ldc], c1);
      Brow += SIMD_FACTOR;
      Cr += SIMD_FACTOR;
    } while (--n);
  }
  if (nRem) {
    int_vector_t mask = pPrm->mask_n[0];
    // partial rightmost word
    fp_vector_t c0 = MM_MASKLOADU_Px(&Cr[0]  , mask);
    fp_vector_t c1 = MM_MASKLOADU_Px(&Cr[ldc], mask);
    fp_vector_t b;

    b = MM_MASKLOADU_Px(&Brow[0], mask);
    c0 = MM_FMADD(b, a00, c0);
    c1 = MM_FMADD(b, a10, c1);

    b = MM_MASKLOADU_Px(&Brow[ldb1], mask);
    c0 = MM_FMADD(b, a01, c0);
    c1 = MM_FMADD(b, a11, c1);

    b = MM_MASKLOADU_Px(&Brow[ldb2], mask);
    c0 = MM_FMADD(b, a02, c0);
    c1 = MM_FMADD(b, a12, c1);

    b = MM_MASKLOADU_Px(&Brow[ldb3], mask);
    c0 = MM_FMADD(b, a03, c0);
    c1 = MM_FMADD(b, a13, c1);

    b = MM_MASKLOADU_Px(&Brow[ldb4], mask);
    c0 = MM_FMADD(b, a04, c0);
    c1 = MM_FMADD(b, a14, c1);

    MM_MASKSTOREU_Px(&Cr[0],   mask, c0);
    MM_MASKSTOREU_Px(&Cr[ldc], mask, c1);
  }
}

static void recursive_saxpy_gemm(const gemm_prm_t* pPrm, unsigned m0, unsigned M, unsigned k0, unsigned K)
{
  int N        = pPrm->N;
  if ((M+K)*N > 32*1024/sizeof(scalar_t)) {
    if (M > K) {
      if (M > 7) {
        unsigned deltaM_a = (M/4)*2;
        recursive_saxpy_gemm(pPrm, m0,          deltaM_a,   k0, K);
        recursive_saxpy_gemm(pPrm, m0+deltaM_a, M-deltaM_a, k0, K);
        return;
      }
    } else {
      if (K > 9) {
        unsigned deltaK_a = (K/10)*5;
        recursive_saxpy_gemm(pPrm, m0, M, k0,          deltaK_a);
        recursive_saxpy_gemm(pPrm, m0, M, k0+deltaK_a, K-deltaK_a);
        return;
      }
    }
  }

  int mLoopCnt = M / 2;
  int mRem     = M % 2;
  int nLoopCnt = N / SIMD_FACTOR;
  int nRem     = N % SIMD_FACTOR;

  const int ldb1 = pPrm->ldb;
  const int ldb2 = ldb1+ldb1;
  const int ldb3 = ldb2+ldb1;
  const int ldb4 = ldb3+ldb1;
  const int ldb5 = ldb4+ldb1;
  const int ldc  = pPrm->ldc;
  const scalar_t *A = pPrm->A + pPrm->lda*m0 + k0;
  const scalar_t *B = pPrm->B + ldb1*k0;
  scalar_t       *C = pPrm->C + ldc*m0;
  K += k0;
  unsigned k;
  for (k = k0; k+5-1 < K; A += 5, B += ldb5, k += 5) {
    scalar_t *Crow = C;
    const scalar_t *Arow = A;
    for (int m = mLoopCnt; m != 0; Crow += ldc*2, --m) {
      if (k == 0) {
        if (pPrm->beta == 0) {
          memset(Crow,     0, N*sizeof(*C));
          memset(Crow+ldc, 0, N*sizeof(*C));
        }
      }
      //saxpy_5x2(pPrm, Arow, B, Crow);
#if 1
      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
      fp_vector_t a00 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[0]), alpha_ps);
      fp_vector_t a01 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[1]), alpha_ps);
      fp_vector_t a02 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[2]), alpha_ps);
      fp_vector_t a03 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[3]), alpha_ps);
      fp_vector_t a04 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[4]), alpha_ps);
      Arow += pPrm->lda;
      fp_vector_t a10 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[0]), alpha_ps);
      fp_vector_t a11 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[1]), alpha_ps);
      fp_vector_t a12 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[2]), alpha_ps);
      fp_vector_t a13 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[3]), alpha_ps);
      fp_vector_t a14 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[4]), alpha_ps);
      Arow += pPrm->lda;
      const scalar_t *Brow = B;
      scalar_t *Cr = Crow;
      if (nLoopCnt) {
        int n = nLoopCnt;
        do {
          fp_vector_t c0 = MM_LOADU_Px(&Cr[0]);
          fp_vector_t c1 = MM_LOADU_Px(&Cr[ldc]);
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

          b = MM_LOADU_Px(&Brow[ldb3]);
          c0 = MM_FMADD(b, a03, c0);
          c1 = MM_FMADD(b, a13, c1);

          b = MM_LOADU_Px(&Brow[ldb4]);
          c0 = MM_FMADD(b, a04, c0);
          c1 = MM_FMADD(b, a14, c1);

          MM_STOREU_Px(&Cr[0],   c0);
          MM_STOREU_Px(&Cr[ldc], c1);
          Brow += SIMD_FACTOR;
          Cr += SIMD_FACTOR;
        } while (--n);
      }
      if (nRem) {
        int_vector_t mask = pPrm->mask_n[0];
        // partial rightmost word
        fp_vector_t c0 = MM_MASKLOADU_Px(&Cr[0]  , mask);
        fp_vector_t c1 = MM_MASKLOADU_Px(&Cr[ldc], mask);
        fp_vector_t b;

        b = MM_MASKLOADU_Px(&Brow[0], mask);
        c0 = MM_FMADD(b, a00, c0);
        c1 = MM_FMADD(b, a10, c1);

        b = MM_MASKLOADU_Px(&Brow[ldb1], mask);
        c0 = MM_FMADD(b, a01, c0);
        c1 = MM_FMADD(b, a11, c1);

        b = MM_MASKLOADU_Px(&Brow[ldb2], mask);
        c0 = MM_FMADD(b, a02, c0);
        c1 = MM_FMADD(b, a12, c1);

        b = MM_MASKLOADU_Px(&Brow[ldb3], mask);
        c0 = MM_FMADD(b, a03, c0);
        c1 = MM_FMADD(b, a13, c1);

        b = MM_MASKLOADU_Px(&Brow[ldb4], mask);
        c0 = MM_FMADD(b, a04, c0);
        c1 = MM_FMADD(b, a14, c1);

        MM_MASKSTOREU_Px(&Cr[0],   mask, c0);
        MM_MASKSTOREU_Px(&Cr[ldc], mask, c1);
      }
#endif
    }
    if (mRem) {
      // bottom row of A and C
      if (k == 0) {
        if (pPrm->beta == 0) {
          memset(Crow, 0, N*sizeof(*C));
        }
      }
      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
      fp_vector_t a00 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[0]), alpha_ps);
      fp_vector_t a01 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[1]), alpha_ps);
      fp_vector_t a02 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[2]), alpha_ps);
      fp_vector_t a03 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[3]), alpha_ps);
      fp_vector_t a04 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[4]), alpha_ps);
      const scalar_t *Brow = B;
      scalar_t *Cr = Crow;
      for (int n = nLoopCnt; n != 0; --n) {
        fp_vector_t c0 = MM_LOADU_Px(&Cr[0]);
        fp_vector_t b;

        b = MM_LOADU_Px(&Brow[0]);
        c0 = MM_FMADD(b, a00, c0);

        b = MM_LOADU_Px(&Brow[ldb1]);
        c0 = MM_FMADD(b, a01, c0);

        b = MM_LOADU_Px(&Brow[ldb2]);
        c0 = MM_FMADD(b, a02, c0);

        b = MM_LOADU_Px(&Brow[ldb3]);
        c0 = MM_FMADD(b, a03, c0);

        b = MM_LOADU_Px(&Brow[ldb4]);
        c0 = MM_FMADD(b, a04, c0);

        MM_STOREU_Px(&Cr[0],   c0);
        Brow += SIMD_FACTOR;
        Cr += SIMD_FACTOR;
      }
      if (nRem) {
        int_vector_t mask = pPrm->mask_n[0];
        // partial rightmost word
        fp_vector_t c0 = MM_MASKLOADU_Px(&Cr[0]  , mask);
        fp_vector_t b;

        b = MM_MASKLOADU_Px(&Brow[0], mask);
        c0 = MM_FMADD(b, a00, c0);

        b = MM_MASKLOADU_Px(&Brow[ldb1], mask);
        c0 = MM_FMADD(b, a01, c0);

        b = MM_MASKLOADU_Px(&Brow[ldb2], mask);
        c0 = MM_FMADD(b, a02, c0);

        b = MM_MASKLOADU_Px(&Brow[ldb3], mask);
        c0 = MM_FMADD(b, a03, c0);

        b = MM_MASKLOADU_Px(&Brow[ldb4], mask);
        c0 = MM_FMADD(b, a04, c0);

        MM_MASKSTOREU_Px(&Cr[0],   mask, c0);
      }
    }
  }

  int remK = K - k;
  if (remK == 4) {
    scalar_t *Crow = C;
    const scalar_t *Arow = A;
    for (int m = mLoopCnt; m != 0; Crow += ldc*2, --m) {
      if (k == 0) {
        if (pPrm->beta == 0) {
          memset(Crow,     0, N*sizeof(*C));
          memset(Crow+ldc, 0, N*sizeof(*C));
        }
      }
      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
      fp_vector_t a00 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[0]), alpha_ps);
      fp_vector_t a01 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[1]), alpha_ps);
      fp_vector_t a02 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[2]), alpha_ps);
      fp_vector_t a03 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[3]), alpha_ps);
      Arow += pPrm->lda;
      fp_vector_t a10 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[0]), alpha_ps);
      fp_vector_t a11 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[1]), alpha_ps);
      fp_vector_t a12 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[2]), alpha_ps);
      fp_vector_t a13 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[3]), alpha_ps);
      Arow += pPrm->lda;
      const scalar_t *Brow = B;
      scalar_t *Cr = Crow;
      for (int n = nLoopCnt; n != 0; --n) {
        fp_vector_t c0 = MM_LOADU_Px(&Cr[0]);
        fp_vector_t c1 = MM_LOADU_Px(&Cr[ldc]);
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

        b = MM_LOADU_Px(&Brow[ldb3]);
        c0 = MM_FMADD(b, a03, c0);
        c1 = MM_FMADD(b, a13, c1);

        MM_STOREU_Px(&Cr[0],   c0);
        MM_STOREU_Px(&Cr[ldc], c1);
        Brow += SIMD_FACTOR;
        Cr += SIMD_FACTOR;
      }
      if (nRem) {
        int_vector_t mask = pPrm->mask_n[0];
        // partial rightmost word
        fp_vector_t c0 = MM_MASKLOADU_Px(&Cr[0]  , mask);
        fp_vector_t c1 = MM_MASKLOADU_Px(&Cr[ldc], mask);
        fp_vector_t b;

        b = MM_MASKLOADU_Px(&Brow[0], mask);
        c0 = MM_FMADD(b, a00, c0);
        c1 = MM_FMADD(b, a10, c1);

        b = MM_MASKLOADU_Px(&Brow[ldb1], mask);
        c0 = MM_FMADD(b, a01, c0);
        c1 = MM_FMADD(b, a11, c1);

        b = MM_MASKLOADU_Px(&Brow[ldb2], mask);
        c0 = MM_FMADD(b, a02, c0);
        c1 = MM_FMADD(b, a12, c1);

        b = MM_MASKLOADU_Px(&Brow[ldb3], mask);
        c0 = MM_FMADD(b, a03, c0);
        c1 = MM_FMADD(b, a13, c1);

        MM_MASKSTOREU_Px(&Cr[0],   mask, c0);
        MM_MASKSTOREU_Px(&Cr[ldc], mask, c1);
      }
    }
    if (mRem) {
      // bottom row of A and C
      if (k == 0) {
        if (pPrm->beta == 0) {
          memset(Crow, 0, N*sizeof(*C));
        }
      }
      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
      fp_vector_t a00 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[0]), alpha_ps);
      fp_vector_t a01 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[1]), alpha_ps);
      fp_vector_t a02 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[2]), alpha_ps);
      fp_vector_t a03 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[3]), alpha_ps);
      const scalar_t *Brow = B;
      scalar_t *Cr = Crow;
      for (int n = nLoopCnt; n != 0; --n) {
        fp_vector_t c0 = MM_LOADU_Px(&Cr[0]);
        fp_vector_t b;

        b = MM_LOADU_Px(&Brow[0]);
        c0 = MM_FMADD(b, a00, c0);

        b = MM_LOADU_Px(&Brow[ldb1]);
        c0 = MM_FMADD(b, a01, c0);

        b = MM_LOADU_Px(&Brow[ldb2]);
        c0 = MM_FMADD(b, a02, c0);

        b = MM_LOADU_Px(&Brow[ldb3]);
        c0 = MM_FMADD(b, a03, c0);

        MM_STOREU_Px(&Cr[0],   c0);
        Brow += SIMD_FACTOR;
        Cr += SIMD_FACTOR;
      }
      if (nRem) {
        int_vector_t mask = pPrm->mask_n[0];
        // partial rightmost word
        fp_vector_t c0 = MM_MASKLOADU_Px(&Cr[0]  , mask);
        fp_vector_t b;

        b = MM_MASKLOADU_Px(&Brow[0], mask);
        c0 = MM_FMADD(b, a00, c0);

        b = MM_MASKLOADU_Px(&Brow[ldb1], mask);
        c0 = MM_FMADD(b, a01, c0);

        b = MM_MASKLOADU_Px(&Brow[ldb2], mask);
        c0 = MM_FMADD(b, a02, c0);

        b = MM_MASKLOADU_Px(&Brow[ldb3], mask);
        c0 = MM_FMADD(b, a03, c0);

        MM_MASKSTOREU_Px(&Cr[0],   mask, c0);
      }
    }
  } else if (remK == 3) {
    scalar_t *Crow = C;
    const scalar_t *Arow = A;
    for (int m = mLoopCnt; m != 0; Crow += ldc*2, --m) {
      if (k == 0) {
        if (pPrm->beta == 0) {
          memset(Crow,     0, N*sizeof(*C));
          memset(Crow+ldc, 0, N*sizeof(*C));
        }
      }
      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
      fp_vector_t a00 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[0]), alpha_ps);
      fp_vector_t a01 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[1]), alpha_ps);
      fp_vector_t a02 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[2]), alpha_ps);
      Arow += pPrm->lda;
      fp_vector_t a10 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[0]), alpha_ps);
      fp_vector_t a11 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[1]), alpha_ps);
      fp_vector_t a12 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[2]), alpha_ps);
      Arow += pPrm->lda;
      const scalar_t *Brow = B;
      scalar_t *Cr = Crow;
      for (int n = nLoopCnt; n != 0; --n) {
        fp_vector_t c0 = MM_LOADU_Px(&Cr[0]);
        fp_vector_t c1 = MM_LOADU_Px(&Cr[ldc]);
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

        MM_STOREU_Px(&Cr[0],   c0);
        MM_STOREU_Px(&Cr[ldc], c1);
        Brow += SIMD_FACTOR;
        Cr += SIMD_FACTOR;
      }
      if (nRem) {
        int_vector_t mask = pPrm->mask_n[0];
        // partial rightmost word
        fp_vector_t c0 = MM_MASKLOADU_Px(&Cr[0]  , mask);
        fp_vector_t c1 = MM_MASKLOADU_Px(&Cr[ldc], mask);
        fp_vector_t b;

        b = MM_MASKLOADU_Px(&Brow[0], mask);
        c0 = MM_FMADD(b, a00, c0);
        c1 = MM_FMADD(b, a10, c1);

        b = MM_MASKLOADU_Px(&Brow[ldb1], mask);
        c0 = MM_FMADD(b, a01, c0);
        c1 = MM_FMADD(b, a11, c1);

        b = MM_MASKLOADU_Px(&Brow[ldb2], mask);
        c0 = MM_FMADD(b, a02, c0);
        c1 = MM_FMADD(b, a12, c1);

        MM_MASKSTOREU_Px(&Cr[0],   mask, c0);
        MM_MASKSTOREU_Px(&Cr[ldc], mask, c1);
      }
    }
    if (mRem) {
      // bottom row of A and C
      if (k == 0) {
        if (pPrm->beta == 0) {
          memset(Crow, 0, N*sizeof(*C));
        }
      }
      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
      fp_vector_t a00 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[0]), alpha_ps);
      fp_vector_t a01 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[1]), alpha_ps);
      fp_vector_t a02 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[2]), alpha_ps);
      const scalar_t *Brow = B;
      scalar_t *Cr = Crow;
      for (int n = nLoopCnt; n != 0; --n) {
        fp_vector_t c0 = MM_LOADU_Px(&Cr[0]);
        fp_vector_t b;

        b = MM_LOADU_Px(&Brow[0]);
        c0 = MM_FMADD(b, a00, c0);

        b = MM_LOADU_Px(&Brow[ldb1]);
        c0 = MM_FMADD(b, a01, c0);

        b = MM_LOADU_Px(&Brow[ldb2]);
        c0 = MM_FMADD(b, a02, c0);

        MM_STOREU_Px(&Cr[0],   c0);
        Brow += SIMD_FACTOR;
        Cr += SIMD_FACTOR;
      }
      if (nRem) {
        int_vector_t mask = pPrm->mask_n[0];
        // partial rightmost word
        fp_vector_t c0 = MM_MASKLOADU_Px(&Cr[0]  , mask);
        fp_vector_t b;

        b = MM_MASKLOADU_Px(&Brow[0], mask);
        c0 = MM_FMADD(b, a00, c0);

        b = MM_MASKLOADU_Px(&Brow[ldb1], mask);
        c0 = MM_FMADD(b, a01, c0);

        b = MM_MASKLOADU_Px(&Brow[ldb2], mask);
        c0 = MM_FMADD(b, a02, c0);

        MM_MASKSTOREU_Px(&Cr[0],   mask, c0);
      }
    }
  } else if (remK == 2) {
    scalar_t *Crow = C;
    const scalar_t *Arow = A;
    for (int m = mLoopCnt; m != 0; Crow += ldc*2, --m) {
      if (k == 0) {
        if (pPrm->beta == 0) {
          memset(Crow,     0, N*sizeof(*C));
          memset(Crow+ldc, 0, N*sizeof(*C));
        }
      }
      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
      fp_vector_t a00 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[0]), alpha_ps);
      fp_vector_t a01 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[1]), alpha_ps);
      Arow += pPrm->lda;
      fp_vector_t a10 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[0]), alpha_ps);
      fp_vector_t a11 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[1]), alpha_ps);
      Arow += pPrm->lda;
      const scalar_t *Brow = B;
      scalar_t *Cr = Crow;
      for (int n = nLoopCnt; n != 0; --n) {
        fp_vector_t c0 = MM_LOADU_Px(&Cr[0]);
        fp_vector_t c1 = MM_LOADU_Px(&Cr[ldc]);
        fp_vector_t b;

        b = MM_LOADU_Px(&Brow[0]);
        c0 = MM_FMADD(b, a00, c0);
        c1 = MM_FMADD(b, a10, c1);

        b = MM_LOADU_Px(&Brow[ldb1]);
        c0 = MM_FMADD(b, a01, c0);
        c1 = MM_FMADD(b, a11, c1);

        MM_STOREU_Px(&Cr[0],   c0);
        MM_STOREU_Px(&Cr[ldc], c1);
        Brow += SIMD_FACTOR;
        Cr += SIMD_FACTOR;
      }
      if (nRem) {
        int_vector_t mask = pPrm->mask_n[0];
        // partial rightmost word
        fp_vector_t c0 = MM_MASKLOADU_Px(&Cr[0]  , mask);
        fp_vector_t c1 = MM_MASKLOADU_Px(&Cr[ldc], mask);
        fp_vector_t b;

        b = MM_MASKLOADU_Px(&Brow[0], mask);
        c0 = MM_FMADD(b, a00, c0);
        c1 = MM_FMADD(b, a10, c1);

        b = MM_MASKLOADU_Px(&Brow[ldb1], mask);
        c0 = MM_FMADD(b, a01, c0);
        c1 = MM_FMADD(b, a11, c1);

        MM_MASKSTOREU_Px(&Cr[0],   mask, c0);
        MM_MASKSTOREU_Px(&Cr[ldc], mask, c1);
      }
    }
    if (mRem) {
      // bottom row of A and C
      if (k == 0) {
        if (pPrm->beta == 0) {
          memset(Crow, 0, N*sizeof(*C));
        }
      }
      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
      fp_vector_t a00 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[0]), alpha_ps);
      fp_vector_t a01 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[1]), alpha_ps);
      const scalar_t *Brow = B;
      scalar_t *Cr = Crow;
      for (int n = nLoopCnt; n != 0; --n) {
        fp_vector_t c0 = MM_LOADU_Px(&Cr[0]);
        fp_vector_t b;

        b = MM_LOADU_Px(&Brow[0]);
        c0 = MM_FMADD(b, a00, c0);

        b = MM_LOADU_Px(&Brow[ldb1]);
        c0 = MM_FMADD(b, a01, c0);

        MM_STOREU_Px(&Cr[0],   c0);
        Brow += SIMD_FACTOR;
        Cr += SIMD_FACTOR;
      }
      if (nRem) {
        int_vector_t mask = pPrm->mask_n[0];
        // partial rightmost word
        fp_vector_t c0 = MM_MASKLOADU_Px(&Cr[0]  , mask);
        fp_vector_t b;

        b = MM_MASKLOADU_Px(&Brow[0], mask);
        c0 = MM_FMADD(b, a00, c0);

        b = MM_MASKLOADU_Px(&Brow[ldb1], mask);
        c0 = MM_FMADD(b, a01, c0);

        MM_MASKSTOREU_Px(&Cr[0],   mask, c0);
      }
    }
  } else if (remK == 1) {
    scalar_t *Crow = C;
    const scalar_t *Arow = A;
    for (int m = mLoopCnt; m != 0; Crow += ldc*2, --m) {
      if (k == 0) {
        if (pPrm->beta == 0) {
          memset(Crow,     0, N*sizeof(*C));
          memset(Crow+ldc, 0, N*sizeof(*C));
        }
      }
      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
      fp_vector_t a00 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[0]), alpha_ps);
      Arow += pPrm->lda;
      fp_vector_t a10 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[0]), alpha_ps);
      Arow += pPrm->lda;
      const scalar_t *Brow = B;
      scalar_t *Cr = Crow;
      for (int n = nLoopCnt; n != 0; --n) {
        fp_vector_t c0 = MM_LOADU_Px(&Cr[0]);
        fp_vector_t c1 = MM_LOADU_Px(&Cr[ldc]);
        fp_vector_t b;

        b = MM_LOADU_Px(&Brow[0]);
        c0 = MM_FMADD(b, a00, c0);
        c1 = MM_FMADD(b, a10, c1);

        MM_STOREU_Px(&Cr[0],   c0);
        MM_STOREU_Px(&Cr[ldc], c1);
        Brow += SIMD_FACTOR;
        Cr += SIMD_FACTOR;
      }
      if (nRem) {
        int_vector_t mask = pPrm->mask_n[0];
        // partial rightmost word
        fp_vector_t c0 = MM_MASKLOADU_Px(&Cr[0]  , mask);
        fp_vector_t c1 = MM_MASKLOADU_Px(&Cr[ldc], mask);
        fp_vector_t b;

        b = MM_MASKLOADU_Px(&Brow[0], mask);
        c0 = MM_FMADD(b, a00, c0);
        c1 = MM_FMADD(b, a10, c1);

        MM_MASKSTOREU_Px(&Cr[0],   mask, c0);
        MM_MASKSTOREU_Px(&Cr[ldc], mask, c1);
      }
    }
    if (mRem) {
      // bottom row of A and C
      if (k == 0) {
        if (pPrm->beta == 0) {
          memset(Crow, 0, N*sizeof(*C));
        }
      }
      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
      fp_vector_t a00 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[0]), alpha_ps);
      const scalar_t *Brow = B;
      scalar_t *Cr = Crow;
      for (int n = nLoopCnt; n != 0; --n) {
        fp_vector_t c0 = MM_LOADU_Px(&Cr[0]);
        fp_vector_t b;

        b = MM_LOADU_Px(&Brow[0]);
        c0 = MM_FMADD(b, a00, c0);

        MM_STOREU_Px(&Cr[0],   c0);
        Brow += SIMD_FACTOR;
        Cr += SIMD_FACTOR;
      }
      if (nRem) {
        int_vector_t mask = pPrm->mask_n[0];
        // partial rightmost word
        fp_vector_t c0 = MM_MASKLOADU_Px(&Cr[0]  , mask);
        fp_vector_t b;

        b = MM_MASKLOADU_Px(&Brow[0], mask);
        c0 = MM_FMADD(b, a00, c0);

        MM_MASKSTOREU_Px(&Cr[0],   mask, c0);
      }
    }
  }
}

void func_name(
  int M, int N, int K,
  scalar_t alpha,
  const scalar_t *A, int lda,
  const scalar_t *B, int ldb,
  scalar_t beta,
  scalar_t *C, int ldc)
{
  if (M < 1 || N < 1 || K < 1)
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

  memset(&prm.mask_n[0], 0, sizeof(prm.mask_n[0]));
  int nRem = N % SIMD_FACTOR;
  if (nRem > 0) // mask on elements of rightmost SIMD word in B and C
    memset((char*)&prm.mask_n[0], -1, sizeof(*C)*nRem);

  recursive_saxpy_gemm(&prm, 0, M, 0, K);
}