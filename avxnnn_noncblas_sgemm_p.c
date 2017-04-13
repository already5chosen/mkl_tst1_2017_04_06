enum {
 SIMD_FACTOR          = sizeof(fp_vector_t)/sizeof(scalar_t),
};

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

  int nLoopCnt = N / SIMD_FACTOR;
  int nRem     = N % SIMD_FACTOR;
  int mLoopCnt = M / 2;
  int mRem     = M % 2;

  int_vector_t mask_n[1];
  memset(&mask_n[0], 0, sizeof(mask_n[0]));
  if (nRem > 0) // mask on elements of rightmost SIMD word in B and C
    memset((char*)&mask_n[0], -1, sizeof(*C)*nRem);

  const int ldb2 = ldb+ldb;
  const int ldb3 = ldb2+ldb;
  const int ldb4 = ldb3+ldb;
  int k;
  for (k = 0; k < K-5+1; A += 5, B += ldb*5, k += 5) {
    scalar_t *Crow = C;
    const scalar_t *Arow = A;
    for (int m = mLoopCnt; m != 0; Crow += ldc*2, --m) {
      if (k == 0) {
        if (beta == 0) {
          memset(Crow,     0, N*sizeof(*C));
          memset(Crow+ldc, 0, N*sizeof(*C));
        }
      }
      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&alpha);
      fp_vector_t a00 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[0]), alpha_ps);
      fp_vector_t a01 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[1]), alpha_ps);
      fp_vector_t a02 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[2]), alpha_ps);
      fp_vector_t a03 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[3]), alpha_ps);
      fp_vector_t a04 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[4]), alpha_ps);
      Arow += lda;
      fp_vector_t a10 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[0]), alpha_ps);
      fp_vector_t a11 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[1]), alpha_ps);
      fp_vector_t a12 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[2]), alpha_ps);
      fp_vector_t a13 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[3]), alpha_ps);
      fp_vector_t a14 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[4]), alpha_ps);
      Arow += lda;
      const scalar_t *Brow = B;
      scalar_t *Cr = Crow;
      for (int n = nLoopCnt; n != 0; --n) {
        fp_vector_t c0 = MM_LOADU_Px(&Cr[0]);
        fp_vector_t c1 = MM_LOADU_Px(&Cr[ldc]);
        fp_vector_t b;

        b = MM_LOADU_Px(&Brow[0]);
        c0 = MM_FMADD(b, a00, c0);
        c1 = MM_FMADD(b, a10, c1);

        b = MM_LOADU_Px(&Brow[ldb]);
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
      }
      if (nRem) {
        int_vector_t mask = mask_n[0];
        // partial rightmost word
        fp_vector_t c0 = MM_MASKLOADU_Px(&Cr[0]  , mask);
        fp_vector_t c1 = MM_MASKLOADU_Px(&Cr[ldc], mask);
        fp_vector_t b;

        b = MM_MASKLOADU_Px(&Brow[0], mask);
        c0 = MM_FMADD(b, a00, c0);
        c1 = MM_FMADD(b, a10, c1);

        b = MM_MASKLOADU_Px(&Brow[ldb], mask);
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
    if (mRem) {
      // bottom row of A and C
      if (k == 0) {
        if (beta == 0) {
          memset(Crow, 0, N*sizeof(*C));
        }
      }
      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&alpha);
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

        b = MM_LOADU_Px(&Brow[ldb]);
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
        int_vector_t mask = mask_n[0];
        // partial rightmost word
        fp_vector_t c0 = MM_MASKLOADU_Px(&Cr[0]  , mask);
        fp_vector_t b;

        b = MM_MASKLOADU_Px(&Brow[0], mask);
        c0 = MM_FMADD(b, a00, c0);

        b = MM_MASKLOADU_Px(&Brow[ldb], mask);
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
        if (beta == 0) {
          memset(Crow,     0, N*sizeof(*C));
          memset(Crow+ldc, 0, N*sizeof(*C));
        }
      }
      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&alpha);
      fp_vector_t a00 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[0]), alpha_ps);
      fp_vector_t a01 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[1]), alpha_ps);
      fp_vector_t a02 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[2]), alpha_ps);
      fp_vector_t a03 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[3]), alpha_ps);
      Arow += lda;
      fp_vector_t a10 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[0]), alpha_ps);
      fp_vector_t a11 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[1]), alpha_ps);
      fp_vector_t a12 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[2]), alpha_ps);
      fp_vector_t a13 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[3]), alpha_ps);
      Arow += lda;
      const scalar_t *Brow = B;
      scalar_t *Cr = Crow;
      for (int n = nLoopCnt; n != 0; --n) {
        fp_vector_t c0 = MM_LOADU_Px(&Cr[0]);
        fp_vector_t c1 = MM_LOADU_Px(&Cr[ldc]);
        fp_vector_t b;

        b = MM_LOADU_Px(&Brow[0]);
        c0 = MM_FMADD(b, a00, c0);
        c1 = MM_FMADD(b, a10, c1);

        b = MM_LOADU_Px(&Brow[ldb]);
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
        int_vector_t mask = mask_n[0];
        // partial rightmost word
        fp_vector_t c0 = MM_MASKLOADU_Px(&Cr[0]  , mask);
        fp_vector_t c1 = MM_MASKLOADU_Px(&Cr[ldc], mask);
        fp_vector_t b;

        b = MM_MASKLOADU_Px(&Brow[0], mask);
        c0 = MM_FMADD(b, a00, c0);
        c1 = MM_FMADD(b, a10, c1);

        b = MM_MASKLOADU_Px(&Brow[ldb], mask);
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
        if (beta == 0) {
          memset(Crow, 0, N*sizeof(*C));
        }
      }
      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&alpha);
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

        b = MM_LOADU_Px(&Brow[ldb]);
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
        int_vector_t mask = mask_n[0];
        // partial rightmost word
        fp_vector_t c0 = MM_MASKLOADU_Px(&Cr[0]  , mask);
        fp_vector_t b;

        b = MM_MASKLOADU_Px(&Brow[0], mask);
        c0 = MM_FMADD(b, a00, c0);

        b = MM_MASKLOADU_Px(&Brow[ldb], mask);
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
        if (beta == 0) {
          memset(Crow,     0, N*sizeof(*C));
          memset(Crow+ldc, 0, N*sizeof(*C));
        }
      }
      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&alpha);
      fp_vector_t a00 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[0]), alpha_ps);
      fp_vector_t a01 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[1]), alpha_ps);
      fp_vector_t a02 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[2]), alpha_ps);
      Arow += lda;
      fp_vector_t a10 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[0]), alpha_ps);
      fp_vector_t a11 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[1]), alpha_ps);
      fp_vector_t a12 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[2]), alpha_ps);
      Arow += lda;
      const scalar_t *Brow = B;
      scalar_t *Cr = Crow;
      for (int n = nLoopCnt; n != 0; --n) {
        fp_vector_t c0 = MM_LOADU_Px(&Cr[0]);
        fp_vector_t c1 = MM_LOADU_Px(&Cr[ldc]);
        fp_vector_t b;

        b = MM_LOADU_Px(&Brow[0]);
        c0 = MM_FMADD(b, a00, c0);
        c1 = MM_FMADD(b, a10, c1);

        b = MM_LOADU_Px(&Brow[ldb]);
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
        int_vector_t mask = mask_n[0];
        // partial rightmost word
        fp_vector_t c0 = MM_MASKLOADU_Px(&Cr[0]  , mask);
        fp_vector_t c1 = MM_MASKLOADU_Px(&Cr[ldc], mask);
        fp_vector_t b;

        b = MM_MASKLOADU_Px(&Brow[0], mask);
        c0 = MM_FMADD(b, a00, c0);
        c1 = MM_FMADD(b, a10, c1);

        b = MM_MASKLOADU_Px(&Brow[ldb], mask);
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
        if (beta == 0) {
          memset(Crow, 0, N*sizeof(*C));
        }
      }
      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&alpha);
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

        b = MM_LOADU_Px(&Brow[ldb]);
        c0 = MM_FMADD(b, a01, c0);

        b = MM_LOADU_Px(&Brow[ldb2]);
        c0 = MM_FMADD(b, a02, c0);

        MM_STOREU_Px(&Cr[0],   c0);
        Brow += SIMD_FACTOR;
        Cr += SIMD_FACTOR;
      }
      if (nRem) {
        int_vector_t mask = mask_n[0];
        // partial rightmost word
        fp_vector_t c0 = MM_MASKLOADU_Px(&Cr[0]  , mask);
        fp_vector_t b;

        b = MM_MASKLOADU_Px(&Brow[0], mask);
        c0 = MM_FMADD(b, a00, c0);

        b = MM_MASKLOADU_Px(&Brow[ldb], mask);
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
        if (beta == 0) {
          memset(Crow,     0, N*sizeof(*C));
          memset(Crow+ldc, 0, N*sizeof(*C));
        }
      }
      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&alpha);
      fp_vector_t a00 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[0]), alpha_ps);
      fp_vector_t a01 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[1]), alpha_ps);
      Arow += lda;
      fp_vector_t a10 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[0]), alpha_ps);
      fp_vector_t a11 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[1]), alpha_ps);
      Arow += lda;
      const scalar_t *Brow = B;
      scalar_t *Cr = Crow;
      for (int n = nLoopCnt; n != 0; --n) {
        fp_vector_t c0 = MM_LOADU_Px(&Cr[0]);
        fp_vector_t c1 = MM_LOADU_Px(&Cr[ldc]);
        fp_vector_t b;

        b = MM_LOADU_Px(&Brow[0]);
        c0 = MM_FMADD(b, a00, c0);
        c1 = MM_FMADD(b, a10, c1);

        b = MM_LOADU_Px(&Brow[ldb]);
        c0 = MM_FMADD(b, a01, c0);
        c1 = MM_FMADD(b, a11, c1);

        MM_STOREU_Px(&Cr[0],   c0);
        MM_STOREU_Px(&Cr[ldc], c1);
        Brow += SIMD_FACTOR;
        Cr += SIMD_FACTOR;
      }
      if (nRem) {
        int_vector_t mask = mask_n[0];
        // partial rightmost word
        fp_vector_t c0 = MM_MASKLOADU_Px(&Cr[0]  , mask);
        fp_vector_t c1 = MM_MASKLOADU_Px(&Cr[ldc], mask);
        fp_vector_t b;

        b = MM_MASKLOADU_Px(&Brow[0], mask);
        c0 = MM_FMADD(b, a00, c0);
        c1 = MM_FMADD(b, a10, c1);

        b = MM_MASKLOADU_Px(&Brow[ldb], mask);
        c0 = MM_FMADD(b, a01, c0);
        c1 = MM_FMADD(b, a11, c1);

        MM_MASKSTOREU_Px(&Cr[0],   mask, c0);
        MM_MASKSTOREU_Px(&Cr[ldc], mask, c1);
      }
    }
    if (mRem) {
      // bottom row of A and C
      if (k == 0) {
        if (beta == 0) {
          memset(Crow, 0, N*sizeof(*C));
        }
      }
      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&alpha);
      fp_vector_t a00 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[0]), alpha_ps);
      fp_vector_t a01 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[1]), alpha_ps);
      const scalar_t *Brow = B;
      scalar_t *Cr = Crow;
      for (int n = nLoopCnt; n != 0; --n) {
        fp_vector_t c0 = MM_LOADU_Px(&Cr[0]);
        fp_vector_t b;

        b = MM_LOADU_Px(&Brow[0]);
        c0 = MM_FMADD(b, a00, c0);

        b = MM_LOADU_Px(&Brow[ldb]);
        c0 = MM_FMADD(b, a01, c0);

        MM_STOREU_Px(&Cr[0],   c0);
        Brow += SIMD_FACTOR;
        Cr += SIMD_FACTOR;
      }
      if (nRem) {
        int_vector_t mask = mask_n[0];
        // partial rightmost word
        fp_vector_t c0 = MM_MASKLOADU_Px(&Cr[0]  , mask);
        fp_vector_t b;

        b = MM_MASKLOADU_Px(&Brow[0], mask);
        c0 = MM_FMADD(b, a00, c0);

        b = MM_MASKLOADU_Px(&Brow[ldb], mask);
        c0 = MM_FMADD(b, a01, c0);

        MM_MASKSTOREU_Px(&Cr[0],   mask, c0);
      }
    }
  } else if (remK == 1) {
    scalar_t *Crow = C;
    const scalar_t *Arow = A;
    for (int m = mLoopCnt; m != 0; Crow += ldc*2, --m) {
      if (k == 0) {
        if (beta == 0) {
          memset(Crow,     0, N*sizeof(*C));
          memset(Crow+ldc, 0, N*sizeof(*C));
        }
      }
      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&alpha);
      fp_vector_t a00 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[0]), alpha_ps);
      Arow += lda;
      fp_vector_t a10 = MM_MUL_Px(MM_BROADCAST_Sx(&Arow[0]), alpha_ps);
      Arow += lda;
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
        int_vector_t mask = mask_n[0];
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
        if (beta == 0) {
          memset(Crow, 0, N*sizeof(*C));
        }
      }
      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&alpha);
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
        int_vector_t mask = mask_n[0];
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
