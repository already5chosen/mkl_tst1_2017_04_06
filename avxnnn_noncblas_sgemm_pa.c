#include <stdio.h>

enum {
 SIMD_FACTOR  = sizeof(fp_vector_t)/sizeof(scalar_t),
};


typedef struct {
  const scalar_t *A;
  fp_vector_t *bb;
  fp_vector_t *cc;
  scalar_t alpha;
  scalar_t beta;
  int lda, ldb, ldc, nw;
  int_vector_t mask_n[1];
  int_vector_t mask_k[1];
} gemm_prm_t;

static void noncblas_sgemm_core(gemm_prm_t* pPrm, unsigned nw, unsigned M, unsigned K)
{
  int mLoopCnt = M / 2;
  int mRem     = M % 2;

  const int ldb1 = pPrm->nw;
  const int ldb2 = ldb1+ldb1;
  const int ldb3 = ldb2+ldb1;
  const int ldb4 = ldb3+ldb1;
  const int ldb5 = ldb4+ldb1;
  const scalar_t *A = pPrm->A;
  const fp_vector_t *B = pPrm->bb;
  unsigned k;
  for (k = 0; k+5-1 < K; A += 5, B += ldb5, k += 5) {
    fp_vector_t *Crow = pPrm->cc;
    const scalar_t *Arow = A;
    for (int m = mLoopCnt; m != 0; Crow += ldb2, --m) {
      fp_vector_t a00 = MM_BROADCAST_Sx(&Arow[0]);
      fp_vector_t a01 = MM_BROADCAST_Sx(&Arow[1]);
      fp_vector_t a02 = MM_BROADCAST_Sx(&Arow[2]);
      fp_vector_t a03 = MM_BROADCAST_Sx(&Arow[3]);
      fp_vector_t a04 = MM_BROADCAST_Sx(&Arow[4]);
      Arow += pPrm->lda;
      fp_vector_t a10 = MM_BROADCAST_Sx(&Arow[0]);
      fp_vector_t a11 = MM_BROADCAST_Sx(&Arow[1]);
      fp_vector_t a12 = MM_BROADCAST_Sx(&Arow[2]);
      fp_vector_t a13 = MM_BROADCAST_Sx(&Arow[3]);
      fp_vector_t a14 = MM_BROADCAST_Sx(&Arow[4]);
      Arow += pPrm->lda;
      const fp_vector_t *Brow = B;
      fp_vector_t *Cr = Crow;
      int n = ldb1;
      do {
        fp_vector_t c0 = Cr[0];
        fp_vector_t c1 = Cr[ldb1];
        fp_vector_t b;

        b = Brow[0];
        c0 = MM_FMADD(b, a00, c0);
        c1 = MM_FMADD(b, a10, c1);

        b = Brow[ldb1];
        c0 = MM_FMADD(b, a01, c0);
        c1 = MM_FMADD(b, a11, c1);

        b = Brow[ldb2];
        c0 = MM_FMADD(b, a02, c0);
        c1 = MM_FMADD(b, a12, c1);

        b = Brow[ldb3];
        c0 = MM_FMADD(b, a03, c0);
        c1 = MM_FMADD(b, a13, c1);

        b = Brow[ldb4];
        c0 = MM_FMADD(b, a04, c0);
        c1 = MM_FMADD(b, a14, c1);

        Cr[0]    = c0;
        Cr[ldb1] = c1;
        Brow += 1;
        Cr   += 1;
      } while (--n);
    }
    if (mRem) {
      // bottom row of A and C
      fp_vector_t a00 = MM_BROADCAST_Sx(&Arow[0]);
      fp_vector_t a01 = MM_BROADCAST_Sx(&Arow[1]);
      fp_vector_t a02 = MM_BROADCAST_Sx(&Arow[2]);
      fp_vector_t a03 = MM_BROADCAST_Sx(&Arow[3]);
      fp_vector_t a04 = MM_BROADCAST_Sx(&Arow[4]);
      const fp_vector_t *Brow = B;
      fp_vector_t *Cr = Crow;
      int n = ldb1;
      do {
        fp_vector_t c0 = Cr[0];
        fp_vector_t c1 = Cr[ldb1];
        fp_vector_t b;

        b = Brow[0];
        c0 = MM_FMADD(b, a00, c0);

        b = Brow[ldb1];
        c0 = MM_FMADD(b, a01, c0);

        b = Brow[ldb2];
        c0 = MM_FMADD(b, a02, c0);

        b = Brow[ldb3];
        c0 = MM_FMADD(b, a03, c0);

        b = Brow[ldb4];
        c0 = MM_FMADD(b, a04, c0);

        Cr[0]    = c0;
        Cr[ldb1] = c1;
        Brow += 1;
        Cr   += 1;
      } while (--n);
    }
  }

  int remK = K - k;
  if (remK==0)
    return;

  if (remK == 4) {
    fp_vector_t *Crow = pPrm->cc;
    const scalar_t *Arow = A;
    for (int m = mLoopCnt; m != 0; Crow += ldb2, --m) {
      fp_vector_t a00 = MM_BROADCAST_Sx(&Arow[0]);
      fp_vector_t a01 = MM_BROADCAST_Sx(&Arow[1]);
      fp_vector_t a02 = MM_BROADCAST_Sx(&Arow[2]);
      fp_vector_t a03 = MM_BROADCAST_Sx(&Arow[3]);
      Arow += pPrm->lda;
      fp_vector_t a10 = MM_BROADCAST_Sx(&Arow[0]);
      fp_vector_t a11 = MM_BROADCAST_Sx(&Arow[1]);
      fp_vector_t a12 = MM_BROADCAST_Sx(&Arow[2]);
      fp_vector_t a13 = MM_BROADCAST_Sx(&Arow[3]);
      Arow += pPrm->lda;
      const fp_vector_t *Brow = B;
      fp_vector_t *Cr = Crow;
      int n = ldb1;
      do {
        fp_vector_t c0 = Cr[0];
        fp_vector_t c1 = Cr[ldb1];
        fp_vector_t b;

        b = Brow[0];
        c0 = MM_FMADD(b, a00, c0);
        c1 = MM_FMADD(b, a10, c1);

        b = Brow[ldb1];
        c0 = MM_FMADD(b, a01, c0);
        c1 = MM_FMADD(b, a11, c1);

        b = Brow[ldb2];
        c0 = MM_FMADD(b, a02, c0);
        c1 = MM_FMADD(b, a12, c1);

        b = Brow[ldb3];
        c0 = MM_FMADD(b, a03, c0);
        c1 = MM_FMADD(b, a13, c1);

        Cr[0]    = c0;
        Cr[ldb1] = c1;
        Brow += 1;
        Cr   += 1;
      } while (--n);
    }
    if (mRem) {
      // bottom row of A and C
      fp_vector_t a00 = MM_BROADCAST_Sx(&Arow[0]);
      fp_vector_t a01 = MM_BROADCAST_Sx(&Arow[1]);
      fp_vector_t a02 = MM_BROADCAST_Sx(&Arow[2]);
      fp_vector_t a03 = MM_BROADCAST_Sx(&Arow[3]);
      const fp_vector_t *Brow = B;
      fp_vector_t *Cr = Crow;
      int n = ldb1;
      do {
        fp_vector_t c0 = Cr[0];
        fp_vector_t c1 = Cr[ldb1];
        fp_vector_t b;

        b = Brow[0];
        c0 = MM_FMADD(b, a00, c0);

        b = Brow[ldb1];
        c0 = MM_FMADD(b, a01, c0);

        b = Brow[ldb2];
        c0 = MM_FMADD(b, a02, c0);

        b = Brow[ldb3];
        c0 = MM_FMADD(b, a03, c0);

        Cr[0]    = c0;
        Cr[ldb1] = c1;
        Brow += 1;
        Cr   += 1;
      } while (--n);
    }
  } else if (remK == 3) {
    fp_vector_t *Crow = pPrm->cc;
    const scalar_t *Arow = A;
    for (int m = mLoopCnt; m != 0; Crow += ldb2, --m) {
      fp_vector_t a00 = MM_BROADCAST_Sx(&Arow[0]);
      fp_vector_t a01 = MM_BROADCAST_Sx(&Arow[1]);
      fp_vector_t a02 = MM_BROADCAST_Sx(&Arow[2]);
      Arow += pPrm->lda;
      fp_vector_t a10 = MM_BROADCAST_Sx(&Arow[0]);
      fp_vector_t a11 = MM_BROADCAST_Sx(&Arow[1]);
      fp_vector_t a12 = MM_BROADCAST_Sx(&Arow[2]);
      Arow += pPrm->lda;
      const fp_vector_t *Brow = B;
      fp_vector_t *Cr = Crow;
      int n = ldb1;
      do {
        fp_vector_t c0 = Cr[0];
        fp_vector_t c1 = Cr[ldb1];
        fp_vector_t b;

        b = Brow[0];
        c0 = MM_FMADD(b, a00, c0);
        c1 = MM_FMADD(b, a10, c1);

        b = Brow[ldb1];
        c0 = MM_FMADD(b, a01, c0);
        c1 = MM_FMADD(b, a11, c1);

        b = Brow[ldb2];
        c0 = MM_FMADD(b, a02, c0);
        c1 = MM_FMADD(b, a12, c1);

        Cr[0]    = c0;
        Cr[ldb1] = c1;
        Brow += 1;
        Cr   += 1;
      } while (--n);
    }
    if (mRem) {
      // bottom row of A and C
      fp_vector_t a00 = MM_BROADCAST_Sx(&Arow[0]);
      fp_vector_t a01 = MM_BROADCAST_Sx(&Arow[1]);
      fp_vector_t a02 = MM_BROADCAST_Sx(&Arow[2]);
      const fp_vector_t *Brow = B;
      fp_vector_t *Cr = Crow;
      int n = ldb1;
      do {
        fp_vector_t c0 = Cr[0];
        fp_vector_t c1 = Cr[ldb1];
        fp_vector_t b;

        b = Brow[0];
        c0 = MM_FMADD(b, a00, c0);

        b = Brow[ldb1];
        c0 = MM_FMADD(b, a01, c0);

        b = Brow[ldb2];
        c0 = MM_FMADD(b, a02, c0);

        Cr[0]    = c0;
        Cr[ldb1] = c1;
        Brow += 1;
        Cr   += 1;
      } while (--n);
    }
  } else if (remK == 2) {
    fp_vector_t *Crow = pPrm->cc;
    const scalar_t *Arow = A;
    for (int m = mLoopCnt; m != 0; Crow += ldb2, --m) {
      fp_vector_t a00 = MM_BROADCAST_Sx(&Arow[0]);
      fp_vector_t a01 = MM_BROADCAST_Sx(&Arow[1]);
      Arow += pPrm->lda;
      fp_vector_t a10 = MM_BROADCAST_Sx(&Arow[0]);
      fp_vector_t a11 = MM_BROADCAST_Sx(&Arow[1]);
      Arow += pPrm->lda;
      const fp_vector_t *Brow = B;
      fp_vector_t *Cr = Crow;
      int n = ldb1;
      do {
        fp_vector_t c0 = Cr[0];
        fp_vector_t c1 = Cr[ldb1];
        fp_vector_t b;

        b = Brow[0];
        c0 = MM_FMADD(b, a00, c0);
        c1 = MM_FMADD(b, a10, c1);

        b = Brow[ldb1];
        c0 = MM_FMADD(b, a01, c0);
        c1 = MM_FMADD(b, a11, c1);

        Cr[0]    = c0;
        Cr[ldb1] = c1;
        Brow += 1;
        Cr   += 1;
      } while (--n);
    }
    if (mRem) {
      // bottom row of A and C
      fp_vector_t a00 = MM_BROADCAST_Sx(&Arow[0]);
      fp_vector_t a01 = MM_BROADCAST_Sx(&Arow[1]);
      const fp_vector_t *Brow = B;
      fp_vector_t *Cr = Crow;
      int n = ldb1;
      do {
        fp_vector_t c0 = Cr[0];
        fp_vector_t c1 = Cr[ldb1];
        fp_vector_t b;

        b = Brow[0];
        c0 = MM_FMADD(b, a00, c0);

        b = Brow[ldb1];
        c0 = MM_FMADD(b, a01, c0);

        Cr[0]    = c0;
        Cr[ldb1] = c1;
        Brow += 1;
        Cr   += 1;
      } while (--n);
    }
  } else { // (remK == 1)
    fp_vector_t *Crow = pPrm->cc;
    const scalar_t *Arow = A;
    for (int m = mLoopCnt; m != 0; Crow += ldb2, --m) {
      fp_vector_t a00 = MM_BROADCAST_Sx(&Arow[0]);
      Arow += pPrm->lda;
      fp_vector_t a10 = MM_BROADCAST_Sx(&Arow[0]);
      Arow += pPrm->lda;
      const fp_vector_t *Brow = B;
      fp_vector_t *Cr = Crow;
      int n = ldb1;
      do {
        fp_vector_t c0 = Cr[0];
        fp_vector_t c1 = Cr[ldb1];
        fp_vector_t b;

        b = Brow[0];
        c0 = MM_FMADD(b, a00, c0);
        c1 = MM_FMADD(b, a10, c1);

        Cr[0]    = c0;
        Cr[ldb1] = c1;
        Brow += 1;
        Cr   += 1;
      } while (--n);
    }
    if (mRem) {
      // bottom row of A and C
      fp_vector_t a00 = MM_BROADCAST_Sx(&Arow[0]);
      const fp_vector_t *Brow = B;
      fp_vector_t *Cr = Crow;
      int n = ldb1;
      do {
        fp_vector_t c0 = Cr[0];
        fp_vector_t c1 = Cr[ldb1];
        fp_vector_t b;

        b = Brow[0];
        c0 = MM_FMADD(b, a00, c0);

        Cr[0]    = c0;
        Cr[ldb1] = c1;
        Brow += 1;
        Cr   += 1;
      } while (--n);
    }
  }
}

static void CopyB(gemm_prm_t* pPrm, unsigned nCol, unsigned nRow, const scalar_t *src)
{
  unsigned nFullCol    = nCol / SIMD_FACTOR;
  unsigned nPartialCol = nCol % SIMD_FACTOR;
  fp_vector_t* dst = pPrm->bb;
  int ldsrc = pPrm->ldb;
  int_vector_t mask = pPrm->mask_n[0];
  do {
    const scalar_t *srcrow = src;
    for (int c = 0; c < nFullCol; ++c) {
      *dst = MM_LOADU_Px(srcrow);
      dst    += 1;
      srcrow += SIMD_FACTOR;
    }
    if (nPartialCol) {
      *dst = MM_MASKLOADU_Px(srcrow, mask);
      dst += 1;
    }
    src += ldsrc;
  } while (--nRow);
}

static void ZeroizeC(fp_vector_t* dst, int cnt)
{
  do {
    *dst = MM_SETZERO_Px();
    ++dst;
  } while (--cnt);
}

static void MultCopyBackC(gemm_prm_t* pPrm, unsigned nCol, unsigned nRow, scalar_t *dst)
{
  unsigned nFullCol    = nCol / SIMD_FACTOR;
  unsigned nPartialCol = nCol % SIMD_FACTOR;
  const fp_vector_t* src = pPrm->cc;
  int lddst = pPrm->ldc;
  int_vector_t mask = pPrm->mask_n[0];
  fp_vector_t alpha_ps  = MM_BROADCAST_Sx(&pPrm->alpha);
  do {
    scalar_t *dstrow = dst;
    for (int c = 0; c < nFullCol; ++c) {
      MM_STOREU_Px(dstrow, MM_MUL_Px(*src, alpha_ps));
      src    += 1;
      dstrow += SIMD_FACTOR;
    }
    if (nPartialCol) {
      MM_MASKSTOREU_Px(dstrow, mask, MM_MUL_Px(*src, alpha_ps));
      src += 1;
    }
    dst += lddst;
  } while (--nRow);
}

static void MaddCopyBackC(gemm_prm_t* pPrm, unsigned nCol, unsigned nRow, scalar_t *dst)
{
  unsigned nFullCol    = nCol / SIMD_FACTOR;
  unsigned nPartialCol = nCol % SIMD_FACTOR;
  const fp_vector_t* src = pPrm->cc;
  int lddst = pPrm->ldc;
  int_vector_t mask = pPrm->mask_n[0];
  fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
  fp_vector_t beta_ps  = MM_BROADCAST_Sx(&pPrm->beta);
  do {
    scalar_t *dstrow = dst;
    for (int c = 0; c < nFullCol; ++c) {
      MM_STOREU_Px(dstrow, MM_FMADD(*src, alpha_ps, MM_MUL_Px(MM_LOADU_Px(dstrow),beta_ps)));
      src    += 1;
      dstrow += SIMD_FACTOR;
    }
    if (nPartialCol) {
      MM_MASKSTOREU_Px(dstrow, mask, MM_FMADD(*src, alpha_ps, MM_MUL_Px(MM_MASKLOADU_Px(dstrow, mask),beta_ps)));
      src += 1;
    }
    dst += lddst;
  } while (--nRow);
}

static unsigned calc_step(unsigned a, unsigned step)
{
  if (a > step+1 && a*2 < step*step) {
    unsigned div = (a-1)/step+1;
    step = (a-1)/div+1;
  }
  unsigned rem0 = (a-1) % (step-1);
  unsigned rem1 = (a-1) % step;
  unsigned rem2 = (a-1) % (step+1);
  if (rem0 > rem1)
    step = (rem0 > rem2) ? step-1 : step+1;
  else
    step = (rem1 > rem2) ? step   : step+1;
  return step < a ? step : a;
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
  prm.alpha = alpha;
  prm.lda   = lda;
  prm.ldb   = ldb;
  prm.beta  = beta;
  prm.ldc   = ldc;

  memset(&prm.mask_n[0], 0, sizeof(prm.mask_n[0]));
  int nRem = (unsigned)N % SIMD_FACTOR;
  if (nRem > 0) // mask on elements of rightmost SIMD word of B and C
    memset((char*)&prm.mask_n[0], -1, sizeof(scalar_t)*nRem);

  unsigned m_step = M_STEP;
  m_step = calc_step((M-1)/2+1, m_step/2)*2;

  unsigned k_step = K_STEP;
  k_step = calc_step((K-1)/5+1, k_step/5)*5;

  unsigned n_step = N_STEP;
  n_step = calc_step((N-1)/SIMD_FACTOR+1, n_step/SIMD_FACTOR)*SIMD_FACTOR;
// printf("m_step=%u k_step=%u n_step=%u\n", m_step, k_step, n_step);

  memset(&prm.mask_k[0], 0, sizeof(prm.mask_k[0]));
  int kRem = (((unsigned)(K-1) % k_step)+1) % SIMD_FACTOR;
  if (kRem > 0) // mask on elements of rightmost SIMD word of A
    memset((char*)&prm.mask_k[0], -1, sizeof(scalar_t)*kRem);

  const int bb_sz = ((n_step-1)/SIMD_FACTOR + 1)*k_step;
  const int cc_sz = ((n_step-1)/SIMD_FACTOR + 1)*m_step;
  const int workBufSz = (bb_sz + cc_sz)*sizeof(fp_vector_t);
  // I didn't find a standard portable way to allocate 32-byte aligned buffer
  // So I am doing it in hackish, but reliable way
  char* workBufAlloc = malloc(workBufSz+sizeof(fp_vector_t));
  uintptr_t workBufAdj = (0-(uintptr_t)(workBufAlloc)) % sizeof(fp_vector_t);
  fp_vector_t* workBuf = (fp_vector_t*)(workBufAlloc+workBufAdj);

  for (int n = 0; n < N; n += n_step) {
    int delta_n = N-n < n_step ? N-n : n_step;
    unsigned nw = (unsigned)(delta_n-1)/SIMD_FACTOR+1;
    prm.nw = nw;
    for (int m = 0; m < M; m += m_step) {
      int delta_m = M-m < m_step ? M-m : m_step;

      scalar_t *Csrcdst = &C[ldc*m+n];
      prm.cc = workBuf;
      ZeroizeC(prm.cc, nw*delta_m);

      const scalar_t *Arow = &A[lda*m];
      prm.bb = prm.cc + nw*delta_m;
      for (int k = 0; k < K; k += k_step) {
        int delta_k = K-k < k_step ? K-k : k_step;
        CopyB(&prm, delta_n, delta_k, &B[ldb*k+n]);
        prm.A = &Arow[k];
        noncblas_sgemm_core(&prm, nw, delta_m, delta_k);
      }

      if (prm.beta == 0.0f) {
        MultCopyBackC(&prm, delta_n, delta_m, Csrcdst);
      } else {
        MaddCopyBackC(&prm, delta_n, delta_m, Csrcdst);
      }
    }
  }
  free(workBufAlloc);
}