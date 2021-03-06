// K < 8
static void noncblas_sgemm_smallK(
  int M, int N, int K,
  scalar_t alpha,
  const scalar_t *A, int lda,
  const scalar_t *B, int ldb,
  scalar_t beta,
  scalar_t *C, int ldc)
{
  int_vector_t Mask_n;
  memset((char*)&Mask_n,  -1, sizeof(Mask_n));
  int nRem = (unsigned)N % SIMD_FACTOR;
  if (nRem > 0) { // mask on elements of rightmost SIMD word of B and C
    memset((char*)&Mask_n,  0, sizeof(Mask_n));
    memset((char*)&Mask_n, -1, sizeof(scalar_t)*nRem);
  }
  // cases of very small K
  fp_vector_t alpha_ps = MM_BROADCAST_Sx(&alpha);
  fp_vector_t beta_ps  = MM_BROADCAST_Sx(&beta);
  int nw   = (unsigned)N / SIMD_FACTOR;
  if (K == 7) {
    const int ldb1 = ldb;
    const int ldb2 = ldb1+ldb1;
    const int ldb3 = ldb2+ldb1;
    const int ldb4 = ldb3+ldb1;
    const int ldb5 = ldb4+ldb1;
    const int ldb6 = ldb5+ldb1;
    for (unsigned mh = (unsigned)M / 2; mh != 0; --mh) {
      const scalar_t *bb = B;
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
        int_vector_t mask_n = Mask_n;

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
    if ((unsigned)M % 2) {
      const scalar_t *bb = B;
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
        int_vector_t mask_n = Mask_n;

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
    const int ldb1 = ldb;
    const int ldb2 = ldb1+ldb1;
    const int ldb3 = ldb2+ldb1;
    const int ldb4 = ldb3+ldb1;
    const int ldb5 = ldb4+ldb1;
    for (unsigned mh = (unsigned)M / 2; mh != 0; --mh) {
      const scalar_t *bb = B;
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
        int_vector_t mask_n = Mask_n;

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
    if ((unsigned)M % 2) {
      const scalar_t *bb = B;
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
        int_vector_t mask_n = Mask_n;

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
    const int ldb1 = ldb;
    const int ldb2 = ldb1+ldb1;
    const int ldb3 = ldb2+ldb1;
    const int ldb4 = ldb3+ldb1;
    for (unsigned mh = (unsigned)M / 2; mh != 0; --mh) {
      const scalar_t *bb = B;
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
        int_vector_t mask_n = Mask_n;

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
    if ((unsigned)M % 2) {
      const scalar_t *bb = B;
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
        int_vector_t mask_n = Mask_n;

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
    const int ldb1 = ldb;
    const int ldb2 = ldb1+ldb1;
    const int ldb3 = ldb2+ldb1;
    for (unsigned mh = (unsigned)M / 2; mh != 0; --mh) {
      const scalar_t *bb = B;
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
        int_vector_t mask_n = Mask_n;

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
    if ((unsigned)M % 2) {
      const scalar_t *bb = B;
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
        int_vector_t mask_n = Mask_n;

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
    const int ldb1 = ldb;
    const int ldb2 = ldb1+ldb1;
    for (unsigned mh = (unsigned)M / 2; mh != 0; --mh) {
      const scalar_t *bb = B;
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
        int_vector_t mask_n = Mask_n;

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
    if ((unsigned)M % 2) {
      const scalar_t *bb = B;
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
        int_vector_t mask_n = Mask_n;

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
    const int ldb1 = ldb;
    for (unsigned mh = (unsigned)M / 2; mh != 0; --mh) {
      const scalar_t *bb = B;
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
        int_vector_t mask_n = Mask_n;

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
    if ((unsigned)M % 2) {
      const scalar_t *bb = B;
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
        int_vector_t mask_n = Mask_n;

        b = MM_MASKLOADU_Px(&bb[0], mask_n);
        fp_vector_t acc0 = MM_MUL_Px(b, a00);

        b = MM_MASKLOADU_Px(&bb[ldb1], mask_n);
        acc0 = MM_FMADD(b, a01, acc0);

        fp_vector_t c0 = MM_MUL_Px(MM_MASKLOADU_Px(&cc[0],   mask_n), beta_ps);
        MM_MASKSTOREU_Px(&cc[0]  , mask_n, MM_FMADD(acc0, alpha_ps, c0));
      }
    }
  } else { // K==1
    for (unsigned mh = (unsigned)M / 2; mh != 0; --mh) {
      const scalar_t *bb = B;
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
        int_vector_t mask_n = Mask_n;

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
    if ((unsigned)M % 2) {
      const scalar_t *bb = B;
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
        int_vector_t mask_n = Mask_n;

        b = MM_MASKLOADU_Px(&bb[0], mask_n);
        fp_vector_t acc0 = MM_MUL_Px(b, a00);

        fp_vector_t c0 = MM_MUL_Px(MM_MASKLOADU_Px(&cc[0],   mask_n), beta_ps);
        MM_MASKSTOREU_Px(&cc[0]  , mask_n, MM_FMADD(acc0, alpha_ps, c0));
      }
    }
  }
}
