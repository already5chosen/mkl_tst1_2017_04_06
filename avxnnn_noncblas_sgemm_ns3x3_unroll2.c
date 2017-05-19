enum {
 SIMD_FACTOR          = sizeof(fp_vector_t)/sizeof(scalar_t),
 A_WORDS_PER_ITER     = 3,
 B_WORDS_PER_ITER     = 3,
 SIMD_ELEM_PEC_COL_MJ = B_WORDS_PER_ITER*N_STEP_MULTIPLIER,
 n_step               = SIMD_ELEM_PEC_COL_MJ*SIMD_FACTOR,
};

enum {
  C_OPTION_UPDATE, C_OPTION_REPLACE, C_OPTION_MULTIPLY
};

typedef struct {
  int           M;
  int           lda;
  int           ldc;
  int           c_option;
  int           masked_b_it;
  scalar_t      alpha;
  scalar_t      beta;
  int_vector_t  mask_n;
  fp_vector_t*  bb;        // [SIMD_ELEM_PEC_COL_MJ*k_step];
  const scalar_t* A;
} noncblas_sgemm_prm_t;

// major core - inner loop processes 3 SIMD columns of B x 3 rows of A
static void fma256_noncblas_sgemm_core_mj(
 const noncblas_sgemm_prm_t* pPrm,
 scalar_t*                   C,
 int                         n_bIters, // 0 < n_bIters <= N_STEP_MULTIPLIER
 int                         nRows)    // 7 < nRows    <= k_step
{
  int ldc = pPrm->ldc;
  int kFullSteps = (unsigned)(nRows-1) / 2;
  int kRemSteps  = (unsigned)(nRows-1) % 2;
  int ldbb   = B_WORDS_PER_ITER*nRows;
  int m;
  const scalar_t* A = pPrm->A;
  int b_itLast = n_bIters - 1;
  int lda1 = pPrm->lda;
  int lda2 = lda1+lda1;
  int lda3 = lda2+lda1;
  for (m = pPrm->M - A_WORDS_PER_ITER + 1; m > 0;
    A += lda3,
    C += ldc*A_WORDS_PER_ITER,
    m -= A_WORDS_PER_ITER) {
    scalar_t* Crow = C;
    #define Prefetch3words(x) \
      _mm_prefetch((char*)(x) + 0,                       _MM_HINT_T0); \
      _mm_prefetch((char*)(x) + sizeof(fp_vector_t)/2*3, _MM_HINT_T0); \
      _mm_prefetch((char*)(x) + sizeof(fp_vector_t)*3-1, _MM_HINT_T0);
    for (int b_it = 0; b_it <= b_itLast; Crow += B_WORDS_PER_ITER*SIMD_FACTOR, ++b_it) {
      const fp_vector_t* Bcol = &pPrm->bb[ldbb*b_it];
      const scalar_t*    ARow = (const scalar_t*)(A);
      scalar_t* CPrefetch = Crow;
      fp_vector_t a;

      fp_vector_t b0 = Bcol[0];
      fp_vector_t b1 = Bcol[1];
      fp_vector_t b2 = Bcol[2];
      Bcol += B_WORDS_PER_ITER;

      a = MM_BROADCAST_Sx(&ARow[0]);
      fp_vector_t acc00 = MM_MUL_Px(a, b0);
      fp_vector_t acc10 = MM_MUL_Px(a, b1);
      fp_vector_t acc20 = MM_MUL_Px(a, b2);
      Prefetch3words(CPrefetch); CPrefetch += ldc;

      a = MM_BROADCAST_Sx(&ARow[lda1+0]);
      fp_vector_t acc01 = MM_MUL_Px(a, b0);
      fp_vector_t acc11 = MM_MUL_Px(a, b1);
      fp_vector_t acc21 = MM_MUL_Px(a, b2);
      Prefetch3words(CPrefetch); CPrefetch += ldc;

      a = MM_BROADCAST_Sx(&ARow[lda2+0]);
      fp_vector_t acc02 = MM_MUL_Px(a, b0);
      fp_vector_t acc12 = MM_MUL_Px(a, b1);
      fp_vector_t acc22 = MM_MUL_Px(a, b2);
      Prefetch3words(CPrefetch); CPrefetch += ldc;

      ARow += 1;

      //__asm volatile("": : :"memory"); // prevent CLANG from screwing register allocation

      int k = kFullSteps;
      do {
        b0 = Bcol[0];
        b1 = Bcol[1];
        b2 = Bcol[2];
        Bcol += B_WORDS_PER_ITER;

        a = MM_BROADCAST_Sx(&ARow[0]);
        acc00 = MM_FMADD(a, b0, acc00);
        acc10 = MM_FMADD(a, b1, acc10);
        acc20 = MM_FMADD(a, b2, acc20);

        a = MM_BROADCAST_Sx(&ARow[lda1+0]);
        acc01 = MM_FMADD(a, b0, acc01);
        acc11 = MM_FMADD(a, b1, acc11);
        acc21 = MM_FMADD(a, b2, acc21);

        a = MM_BROADCAST_Sx(&ARow[lda2+0]);
        acc02 = MM_FMADD(a, b0, acc02);
        acc12 = MM_FMADD(a, b1, acc12);
        acc22 = MM_FMADD(a, b2, acc22);

        b0 = Bcol[0];
        b1 = Bcol[1];
        b2 = Bcol[2];
        Bcol += B_WORDS_PER_ITER;

        a = MM_BROADCAST_Sx(&ARow[1]);
        acc00 = MM_FMADD(a, b0, acc00);
        acc10 = MM_FMADD(a, b1, acc10);
        acc20 = MM_FMADD(a, b2, acc20);

        a = MM_BROADCAST_Sx(&ARow[lda1+1]);
        acc01 = MM_FMADD(a, b0, acc01);
        acc11 = MM_FMADD(a, b1, acc11);
        acc21 = MM_FMADD(a, b2, acc21);

        a = MM_BROADCAST_Sx(&ARow[lda2+1]);
        acc02 = MM_FMADD(a, b0, acc02);
        acc12 = MM_FMADD(a, b1, acc12);
        acc22 = MM_FMADD(a, b2, acc22);

        ARow += 2;
      } while (--k);

      if (kRemSteps != 0) {
        b0 = Bcol[0];
        b1 = Bcol[1];
        b2 = Bcol[2];
        Bcol += B_WORDS_PER_ITER;

        a = MM_BROADCAST_Sx(&ARow[0]);
        acc00 = MM_FMADD(a, b0, acc00);
        acc10 = MM_FMADD(a, b1, acc10);
        acc20 = MM_FMADD(a, b2, acc20);

        a = MM_BROADCAST_Sx(&ARow[lda1+0]);
        acc01 = MM_FMADD(a, b0, acc01);
        acc11 = MM_FMADD(a, b1, acc11);
        acc21 = MM_FMADD(a, b2, acc21);

        a = MM_BROADCAST_Sx(&ARow[lda2+0]);
        acc02 = MM_FMADD(a, b0, acc02);
        acc12 = MM_FMADD(a, b1, acc12);
        acc22 = MM_FMADD(a, b2, acc22);
      }

      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
      scalar_t* CCol = Crow;

      if (b_it != pPrm->masked_b_it) {
        if (pPrm->c_option == C_OPTION_UPDATE) {
          #define UPDATE_CCOL(ccol, acc0, acc1, acc2) \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*0]), MM_FMADD((acc0), alpha_ps, MM_LOADU_Px(&((ccol)[SIMD_FACTOR*0])))); \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*1]), MM_FMADD((acc1), alpha_ps, MM_LOADU_Px(&((ccol)[SIMD_FACTOR*1])))); \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*2]), MM_FMADD((acc2), alpha_ps, MM_LOADU_Px(&((ccol)[SIMD_FACTOR*2]))));

          UPDATE_CCOL(CCol, acc00, acc10, acc20); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11, acc21); CCol += ldc;
          UPDATE_CCOL(CCol, acc02, acc12, acc22);

          #undef UPDATE_CCOL
        } else if (pPrm->c_option == C_OPTION_REPLACE) {
          #define UPDATE_CCOL(ccol, acc0, acc1, acc2) \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*0]), MM_MUL_Px((acc0), alpha_ps)); \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*1]), MM_MUL_Px((acc1), alpha_ps)); \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*2]), MM_MUL_Px((acc2), alpha_ps));

          UPDATE_CCOL(CCol, acc00, acc10, acc20); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11, acc21); CCol += ldc;
          UPDATE_CCOL(CCol, acc02, acc12, acc22);

          #undef UPDATE_CCOL
        } else { // C_OPTION_MULTIPLY
          fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
          #define UPDATE_CCOL(ccol, acc0, acc1, acc2) \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*0]), MM_FMADD((acc0), alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px(&((ccol)[SIMD_FACTOR*0]))))); \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*1]), MM_FMADD((acc1), alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px(&((ccol)[SIMD_FACTOR*1]))))); \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*2]), MM_FMADD((acc2), alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px(&((ccol)[SIMD_FACTOR*2])))));

          UPDATE_CCOL(CCol, acc00, acc10, acc20); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11, acc21); CCol += ldc;
          UPDATE_CCOL(CCol, acc02, acc12, acc22);

          #undef UPDATE_CCOL
        }
      } else {
        int_vector_t mask = pPrm->mask_n;
        if (pPrm->c_option == C_OPTION_UPDATE) {
          #define UPDATE_CCOL(ccol, acc0, acc1, acc2) \
          MM_STOREU_Px    (&((ccol)[SIMD_FACTOR*0]),       MM_FMADD((acc0), alpha_ps, MM_LOADU_Px    (&((ccol)[SIMD_FACTOR*0])))); \
          MM_STOREU_Px    (&((ccol)[SIMD_FACTOR*1]),       MM_FMADD((acc1), alpha_ps, MM_LOADU_Px    (&((ccol)[SIMD_FACTOR*1])))); \
          MM_MASKSTOREU_Px(&((ccol)[SIMD_FACTOR*2]), mask, MM_FMADD((acc2), alpha_ps, MM_MASKLOADU_Px(&((ccol)[SIMD_FACTOR*2]), mask)));

          UPDATE_CCOL(CCol, acc00, acc10, acc20); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11, acc21); CCol += ldc;
          UPDATE_CCOL(CCol, acc02, acc12, acc22);

          #undef UPDATE_CCOL
        } else if (pPrm->c_option == C_OPTION_REPLACE) {
          #define UPDATE_CCOL(ccol, acc0, acc1, acc2) \
          MM_STOREU_Px    (&((ccol)[SIMD_FACTOR*0]),       MM_MUL_Px((acc0), alpha_ps)); \
          MM_STOREU_Px    (&((ccol)[SIMD_FACTOR*1]),       MM_MUL_Px((acc1), alpha_ps)); \
          MM_MASKSTOREU_Px(&((ccol)[SIMD_FACTOR*2]), mask, MM_MUL_Px((acc2), alpha_ps));

          UPDATE_CCOL(CCol, acc00, acc10, acc20); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11, acc21); CCol += ldc;
          UPDATE_CCOL(CCol, acc02, acc12, acc22);

          #undef UPDATE_CCOL
        } else { // C_OPTION_MULTIPLY
          fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
          #define UPDATE_CCOL(ccol, acc0, acc1, acc2) \
          MM_STOREU_Px    (&((ccol)[SIMD_FACTOR*0]),       MM_FMADD((acc0), alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px    (&((ccol)[SIMD_FACTOR*0]))))); \
          MM_STOREU_Px    (&((ccol)[SIMD_FACTOR*1]),       MM_FMADD((acc1), alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px    (&((ccol)[SIMD_FACTOR*1]))))); \
          MM_MASKSTOREU_Px(&((ccol)[SIMD_FACTOR*2]), mask, MM_FMADD((acc2), alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(&((ccol)[SIMD_FACTOR*2]), mask))));

          UPDATE_CCOL(CCol, acc00, acc10, acc20); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11, acc21); CCol += ldc;
          UPDATE_CCOL(CCol, acc02, acc12, acc22);

          #undef UPDATE_CCOL
        }
      }
    }
  }

  // handle remaining rows of A
  m += A_WORDS_PER_ITER - 1;
  kFullSteps = (unsigned)(nRows-1) / 4;
  kRemSteps  = (unsigned)(nRows-1) % 4;
  for (; m > 0;  A += lda1, C += ldc, --m) {
    scalar_t* Crow = C;
    for (int b_it = 0; b_it <= b_itLast; Crow += B_WORDS_PER_ITER*SIMD_FACTOR, ++b_it) {
      const fp_vector_t* Bcol = &pPrm->bb[ldbb*b_it];
      const scalar_t*    ARow = A;
      fp_vector_t a = MM_BROADCAST_Sx(&ARow[0]);
      ARow += 1;
      fp_vector_t acc00 = MM_MUL_Px(a, Bcol[0]);
      fp_vector_t acc10 = MM_MUL_Px(a, Bcol[1]);
      fp_vector_t acc20 = MM_MUL_Px(a, Bcol[2]);
      Bcol += B_WORDS_PER_ITER;

      fp_vector_t acc01 = MM_SETZERO_Px();
      fp_vector_t acc11 = MM_SETZERO_Px();
      fp_vector_t acc21 = MM_SETZERO_Px();

      fp_vector_t acc02 = MM_SETZERO_Px();
      fp_vector_t acc12 = MM_SETZERO_Px();
      fp_vector_t acc22 = MM_SETZERO_Px();

      fp_vector_t acc03 = MM_SETZERO_Px();
      fp_vector_t acc13 = MM_SETZERO_Px();
      fp_vector_t acc23 = MM_SETZERO_Px();

      _mm_prefetch((char*)(Crow)+0,                       _MM_HINT_T0);
      _mm_prefetch((char*)(Crow)+sizeof(fp_vector_t)/2*3, _MM_HINT_T0);
      _mm_prefetch((char*)(Crow)+sizeof(fp_vector_t)*3-1, _MM_HINT_T0);

      int k = kFullSteps;
      do {
        a = MM_BROADCAST_Sx(&ARow[0]);
        acc00 = MM_FMADD(a, Bcol[0], acc00);
        acc10 = MM_FMADD(a, Bcol[1], acc10);
        acc20 = MM_FMADD(a, Bcol[2], acc20);
        Bcol += B_WORDS_PER_ITER;

        a = MM_BROADCAST_Sx(&ARow[1]);
        acc01 = MM_FMADD(a, Bcol[0], acc01);
        acc11 = MM_FMADD(a, Bcol[1], acc11);
        acc21 = MM_FMADD(a, Bcol[2], acc21);
        Bcol += B_WORDS_PER_ITER;

        a = MM_BROADCAST_Sx(&ARow[2]);
        acc02 = MM_FMADD(a, Bcol[0], acc02);
        acc12 = MM_FMADD(a, Bcol[1], acc12);
        acc22 = MM_FMADD(a, Bcol[2], acc22);
        Bcol += B_WORDS_PER_ITER;

        a = MM_BROADCAST_Sx(&ARow[3]);
        acc03 = MM_FMADD(a, Bcol[0], acc03);
        acc13 = MM_FMADD(a, Bcol[1], acc13);
        acc23 = MM_FMADD(a, Bcol[2], acc23);
        Bcol += B_WORDS_PER_ITER;

        ARow += 4;
      } while (--k);

      if (kRemSteps != 0) {
        fp_vector_t a;

        a = MM_BROADCAST_Sx(&ARow[0]);
        acc00 = MM_FMADD(a, Bcol[0], acc00);
        acc10 = MM_FMADD(a, Bcol[1], acc10);
        acc20 = MM_FMADD(a, Bcol[2], acc20);
        if (kRemSteps != 1) {
          Bcol += B_WORDS_PER_ITER;

          a = MM_BROADCAST_Sx(&ARow[1]);
          acc01 = MM_FMADD(a, Bcol[0], acc01);
          acc11 = MM_FMADD(a, Bcol[1], acc11);
          acc21 = MM_FMADD(a, Bcol[2], acc21);
          if (kRemSteps != 2) {
            Bcol += B_WORDS_PER_ITER;

            a = MM_BROADCAST_Sx(&ARow[2]);
            acc02 = MM_FMADD(a, Bcol[0], acc02);
            acc12 = MM_FMADD(a, Bcol[1], acc12);
            acc22 = MM_FMADD(a, Bcol[2], acc22);
          }
        }
      }

      acc00 = MM_ADD_Px(acc00, acc01);
      acc02 = MM_ADD_Px(acc02, acc03);

      acc10 = MM_ADD_Px(acc10, acc11);
      acc12 = MM_ADD_Px(acc12, acc13);

      acc20 = MM_ADD_Px(acc20, acc21);
      acc22 = MM_ADD_Px(acc22, acc23);

      acc00 = MM_ADD_Px(acc00, acc02);
      acc10 = MM_ADD_Px(acc10, acc12);
      acc20 = MM_ADD_Px(acc20, acc22);

      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
      if (b_it != pPrm->masked_b_it) {
        if (pPrm->c_option == C_OPTION_UPDATE) {
          MM_STOREU_Px(&Crow[SIMD_FACTOR*0], MM_FMADD(acc00, alpha_ps, MM_LOADU_Px(&Crow[SIMD_FACTOR*0])));
          MM_STOREU_Px(&Crow[SIMD_FACTOR*1], MM_FMADD(acc10, alpha_ps, MM_LOADU_Px(&Crow[SIMD_FACTOR*1])));
          MM_STOREU_Px(&Crow[SIMD_FACTOR*2], MM_FMADD(acc20, alpha_ps, MM_LOADU_Px(&Crow[SIMD_FACTOR*2])));
        } else if (pPrm->c_option == C_OPTION_REPLACE) {
          MM_STOREU_Px(&Crow[SIMD_FACTOR*0], MM_MUL_Px(acc00, alpha_ps));
          MM_STOREU_Px(&Crow[SIMD_FACTOR*1], MM_MUL_Px(acc10, alpha_ps));
          MM_STOREU_Px(&Crow[SIMD_FACTOR*2], MM_MUL_Px(acc20, alpha_ps));
        } else { // C_OPTION_MULTIPLY
          fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
          MM_STOREU_Px(&Crow[SIMD_FACTOR*0], MM_FMADD(acc00, alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px(&Crow[SIMD_FACTOR*0]))));
          MM_STOREU_Px(&Crow[SIMD_FACTOR*1], MM_FMADD(acc10, alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px(&Crow[SIMD_FACTOR*1]))));
          MM_STOREU_Px(&Crow[SIMD_FACTOR*2], MM_FMADD(acc20, alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px(&Crow[SIMD_FACTOR*2]))));
        }
      } else {
        int_vector_t mask = pPrm->mask_n;
        if (pPrm->c_option == C_OPTION_UPDATE) {
          MM_STOREU_Px    (&Crow[SIMD_FACTOR*0],       MM_FMADD(acc00, alpha_ps, MM_LOADU_Px    (&Crow[SIMD_FACTOR*0])));
          MM_STOREU_Px    (&Crow[SIMD_FACTOR*1],       MM_FMADD(acc10, alpha_ps, MM_LOADU_Px    (&Crow[SIMD_FACTOR*1])));
          MM_MASKSTOREU_Px(&Crow[SIMD_FACTOR*2], mask, MM_FMADD(acc20, alpha_ps, MM_MASKLOADU_Px(&Crow[SIMD_FACTOR*2], mask)));
        } else if (pPrm->c_option == C_OPTION_REPLACE) {
          MM_STOREU_Px    (&Crow[SIMD_FACTOR*0],       MM_MUL_Px(acc00, alpha_ps));
          MM_STOREU_Px    (&Crow[SIMD_FACTOR*1],       MM_MUL_Px(acc10, alpha_ps));
          MM_MASKSTOREU_Px(&Crow[SIMD_FACTOR*2], mask, MM_MUL_Px(acc20, alpha_ps));
        } else { // C_OPTION_MULTIPLY
          fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
          MM_STOREU_Px    (&Crow[SIMD_FACTOR*0],       MM_FMADD(acc00, alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px    (&Crow[SIMD_FACTOR*0]))));
          MM_STOREU_Px    (&Crow[SIMD_FACTOR*1],       MM_FMADD(acc10, alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px    (&Crow[SIMD_FACTOR*1]))));
          MM_MASKSTOREU_Px(&Crow[SIMD_FACTOR*2], mask, MM_FMADD(acc20, alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(&Crow[SIMD_FACTOR*2], mask))));
        }
      }
    }
  }
}

// minor core - inner loop processes 1 SIMD columns of B x 4 rows of A
static void fma256_noncblas_sgemm_core_mn1(
 const noncblas_sgemm_prm_t* pPrm,
 scalar_t*                   C,
 int                         nRows)    // 0 < nRows <= k_step
{
  const int A_WORDS_PER_ITER_MN1 = 4;
  int ldc = pPrm->ldc;
  int kFullSteps = (unsigned)(nRows) / 4;
  int kRemSteps  = (unsigned)(nRows) % 4;
  int m;
  int lda1 = pPrm->lda;
  int lda2 = lda1+lda1;
  int lda3 = lda2+lda1;
  int lda4 = lda3+lda1;
  const scalar_t* A = pPrm->A;
  for (m = 0; m < pPrm->M - A_WORDS_PER_ITER_MN1 + 1; A += lda4, m += A_WORDS_PER_ITER_MN1) {
    const fp_vector_t* Bcol = pPrm->bb;
    fp_vector_t acc00 = MM_SETZERO_Px();
    fp_vector_t acc10 = MM_SETZERO_Px();
    fp_vector_t acc01 = MM_SETZERO_Px();
    fp_vector_t acc11 = MM_SETZERO_Px();
    fp_vector_t acc02 = MM_SETZERO_Px();
    fp_vector_t acc12 = MM_SETZERO_Px();
    fp_vector_t acc03 = MM_SETZERO_Px();
    fp_vector_t acc13 = MM_SETZERO_Px();

    scalar_t* CCol = C;
    for (int ci = 0; ci < A_WORDS_PER_ITER_MN1; ++ci) {
      _mm_prefetch((char*)(CCol), _MM_HINT_T0);
      CCol += ldc;
    }
    const scalar_t* ARow = A;
    for (int k = 0; k < kFullSteps; ++k) {
      fp_vector_t b;

      b = Bcol[0];
      acc00 = MM_FMADD(MM_BROADCAST_Sx(&ARow[0]), b, acc00);
      acc01 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda1 + 0]), b, acc01);
      acc02 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda2 + 0]), b, acc02);
      acc03 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda3 + 0]), b, acc03);

      b = Bcol[1];
      acc10 = MM_FMADD(MM_BROADCAST_Sx(&ARow[1]), b, acc10);
      acc11 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda1 + 1]), b, acc11);
      acc12 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda2 + 1]), b, acc12);
      acc13 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda3 + 1]), b, acc13);

      b = Bcol[2];
      acc00 = MM_FMADD(MM_BROADCAST_Sx(&ARow[2]), b, acc00);
      acc01 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda1 + 2]), b, acc01);
      acc02 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda2 + 2]), b, acc02);
      acc03 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda3 + 2]), b, acc03);

      b = Bcol[3];
      acc10 = MM_FMADD(MM_BROADCAST_Sx(&ARow[3]), b, acc10);
      acc11 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda1 + 3]), b, acc11);
      acc12 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda2 + 3]), b, acc12);
      acc13 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda3 + 3]), b, acc13);

      Bcol += 4;
      ARow += 4;
    }
    if (kRemSteps != 0) {
      fp_vector_t b;

      b = Bcol[0];
      acc00 = MM_FMADD(MM_BROADCAST_Sx(&ARow[0]), b, acc00);
      acc01 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda1 + 0]), b, acc01);
      acc02 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda2 + 0]), b, acc02);
      acc03 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda3 + 0]), b, acc03);

      if (kRemSteps != 1) {
        b = Bcol[1];
        acc10 = MM_FMADD(MM_BROADCAST_Sx(&ARow[1]), b, acc10);
        acc11 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda1 + 1]), b, acc11);
        acc12 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda2 + 1]), b, acc12);
        acc13 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda3 + 1]), b, acc13);

        if (kRemSteps != 2) {
          b = Bcol[2];
          acc00 = MM_FMADD(MM_BROADCAST_Sx(&ARow[2]), b, acc00);
          acc01 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda1 + 2]), b, acc01);
          acc02 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda2 + 2]), b, acc02);
          acc03 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda3 + 2]), b, acc03);
        }
      }
    }
    acc00 = MM_ADD_Px(acc00, acc10);
    acc01 = MM_ADD_Px(acc01, acc11);
    acc02 = MM_ADD_Px(acc02, acc12);
    acc03 = MM_ADD_Px(acc03, acc13);

    fp_vector_t  alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
    int_vector_t mask = pPrm->mask_n;

    if (pPrm->c_option == C_OPTION_UPDATE) {
      MM_MASKSTOREU_Px(C, mask, MM_FMADD(acc00, alpha_ps, MM_MASKLOADU_Px(C, mask))); C += ldc;
      MM_MASKSTOREU_Px(C, mask, MM_FMADD(acc01, alpha_ps, MM_MASKLOADU_Px(C, mask))); C += ldc;
      MM_MASKSTOREU_Px(C, mask, MM_FMADD(acc02, alpha_ps, MM_MASKLOADU_Px(C, mask))); C += ldc;
      MM_MASKSTOREU_Px(C, mask, MM_FMADD(acc03, alpha_ps, MM_MASKLOADU_Px(C, mask))); C += ldc;
    } else if (pPrm->c_option == C_OPTION_REPLACE) {
      MM_MASKSTOREU_Px(C, mask, MM_MUL_Px(acc00, alpha_ps)); C += ldc;
      MM_MASKSTOREU_Px(C, mask, MM_MUL_Px(acc01, alpha_ps)); C += ldc;
      MM_MASKSTOREU_Px(C, mask, MM_MUL_Px(acc02, alpha_ps)); C += ldc;
      MM_MASKSTOREU_Px(C, mask, MM_MUL_Px(acc03, alpha_ps)); C += ldc;
    } else { // C_OPTION_MULTIPLY
      fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
      MM_MASKSTOREU_Px(C, mask, MM_FMADD(acc00, alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(C, mask)))); C += ldc;
      MM_MASKSTOREU_Px(C, mask, MM_FMADD(acc01, alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(C, mask)))); C += ldc;
      MM_MASKSTOREU_Px(C, mask, MM_FMADD(acc02, alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(C, mask)))); C += ldc;
      MM_MASKSTOREU_Px(C, mask, MM_FMADD(acc03, alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(C, mask)))); C += ldc;
    }
  }

  // handle remaining rows of a - non-interleaved
  for (; m < pPrm->M; A += lda1, ++m) {
    const fp_vector_t* Bcol = pPrm->bb;
    fp_vector_t acc00 = MM_SETZERO_Px();
    fp_vector_t acc01 = MM_SETZERO_Px();
    fp_vector_t acc02 = MM_SETZERO_Px();
    fp_vector_t acc03 = MM_SETZERO_Px();

    _mm_prefetch((char*)(C), _MM_HINT_T0);
    const scalar_t* ARow = A;
    for (int k = 0; k < kFullSteps; ++k) {
      acc00 = MM_FMADD(MM_BROADCAST_Sx(&ARow[0]), Bcol[0], acc00);
      acc01 = MM_FMADD(MM_BROADCAST_Sx(&ARow[1]), Bcol[1], acc01);
      acc02 = MM_FMADD(MM_BROADCAST_Sx(&ARow[2]), Bcol[2], acc02);
      acc03 = MM_FMADD(MM_BROADCAST_Sx(&ARow[3]), Bcol[3], acc03);
      Bcol += 4;
      ARow += 4;
    }
    if (kRemSteps != 0) {
      acc00 = MM_FMADD(MM_BROADCAST_Sx(&ARow[0]), Bcol[0], acc00);
      if (kRemSteps != 1) {
        acc01 = MM_FMADD(MM_BROADCAST_Sx(&ARow[1]), Bcol[1], acc01);
        if (kRemSteps != 2) {
          acc02 = MM_FMADD(MM_BROADCAST_Sx(&ARow[2]), Bcol[2], acc02);
        }
      }
    }
    acc00 = MM_ADD_Px(acc00, acc01);
    acc02 = MM_ADD_Px(acc02, acc03);
    acc00 = MM_ADD_Px(acc00, acc02);

    fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
    int_vector_t mask = pPrm->mask_n;
    if (pPrm->c_option == C_OPTION_UPDATE) {
      MM_MASKSTOREU_Px(C, mask, MM_FMADD(acc00, alpha_ps, MM_MASKLOADU_Px(C, mask)));
    } else if (pPrm->c_option == C_OPTION_REPLACE) {
      MM_MASKSTOREU_Px(C, mask, MM_MUL_Px(acc00, alpha_ps));
    } else { // C_OPTION_MULTIPLY
      fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
      MM_MASKSTOREU_Px(C, mask, MM_FMADD(acc00, alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(C, mask))));
    }
    C += ldc;
  }
}

// minor core - inner loop processes 2 SIMD columns of B x 5 rows of A
static void fma256_noncblas_sgemm_core_mn2(
 const noncblas_sgemm_prm_t* pPrm,
 scalar_t*                   C,
 int                         nRows)    // 0 < nRows <= k_step
{
  const int A_WORDS_PER_ITER_MN2 = 5;
  const int B_WORDS_PER_ITER_MN2 = 2;
  int ldc = pPrm->ldc;
  int kFullSteps = (unsigned)(nRows) / 4;
  int kRemSteps  = (unsigned)(nRows) % 4;
  int m;
  int lda1 = pPrm->lda;
  int lda2 = lda1+lda1;
  int lda3 = lda2+lda1;
  int lda4 = lda3+lda1;
  int lda5 = lda4+lda1;
  const scalar_t* A = pPrm->A;
  for (m = pPrm->M - A_WORDS_PER_ITER_MN2 + 1; m > 0; A += lda5, m -= A_WORDS_PER_ITER_MN2) {
    const fp_vector_t* Bcol = pPrm->bb;
    scalar_t* CCol = C;
    fp_vector_t acc00 = MM_SETZERO_Px();
    fp_vector_t acc10 = MM_SETZERO_Px();
    _mm_prefetch((char*)(CCol), _MM_HINT_T0);
    _mm_prefetch((char*)(CCol)+sizeof(fp_vector_t)*2-1, _MM_HINT_T0);CCol += ldc;
    fp_vector_t acc01 = MM_SETZERO_Px();
    fp_vector_t acc11 = MM_SETZERO_Px();
    _mm_prefetch((char*)(CCol), _MM_HINT_T0);
    _mm_prefetch((char*)(CCol)+sizeof(fp_vector_t)*2-1, _MM_HINT_T0);CCol += ldc;
    fp_vector_t acc02 = MM_SETZERO_Px();
    fp_vector_t acc12 = MM_SETZERO_Px();
    _mm_prefetch((char*)(CCol), _MM_HINT_T0);
    _mm_prefetch((char*)(CCol)+sizeof(fp_vector_t)*2-1, _MM_HINT_T0);CCol += ldc;
    fp_vector_t acc03 = MM_SETZERO_Px();
    fp_vector_t acc13 = MM_SETZERO_Px();
    _mm_prefetch((char*)(CCol), _MM_HINT_T0);
    _mm_prefetch((char*)(CCol)+sizeof(fp_vector_t)*2-1, _MM_HINT_T0);CCol += ldc;
    fp_vector_t acc04 = MM_SETZERO_Px();
    fp_vector_t acc14 = MM_SETZERO_Px();
    _mm_prefetch((char*)(CCol), _MM_HINT_T0);
    _mm_prefetch((char*)(CCol)+sizeof(fp_vector_t)*2-1, _MM_HINT_T0);CCol += ldc;

    const scalar_t* ARow = A;
    for (int k = 0; k < kFullSteps; ++k) {
      fp_vector_t a, b0, b1;
      b0 = Bcol[0];
      b1 = Bcol[1];
      Bcol += B_WORDS_PER_ITER_MN2;

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
      Bcol += B_WORDS_PER_ITER_MN2;

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

      b0 = Bcol[0];
      b1 = Bcol[1];
      Bcol += B_WORDS_PER_ITER_MN2;

      a = MM_BROADCAST_Sx(&ARow[2]);
      acc00 = MM_FMADD(a, b0, acc00);
      acc10 = MM_FMADD(a, b1, acc10);

      a = MM_BROADCAST_Sx(&ARow[lda1+2]);
      acc01 = MM_FMADD(a, b0, acc01);
      acc11 = MM_FMADD(a, b1, acc11);

      a = MM_BROADCAST_Sx(&ARow[lda2+2]);
      acc02 = MM_FMADD(a, b0, acc02);
      acc12 = MM_FMADD(a, b1, acc12);

      a = MM_BROADCAST_Sx(&ARow[lda3+2]);
      acc03 = MM_FMADD(a, b0, acc03);
      acc13 = MM_FMADD(a, b1, acc13);

      a = MM_BROADCAST_Sx(&ARow[lda4+2]);
      acc04 = MM_FMADD(a, b0, acc04);
      acc14 = MM_FMADD(a, b1, acc14);

      b0 = Bcol[0];
      b1 = Bcol[1];
      Bcol += B_WORDS_PER_ITER_MN2;

      a = MM_BROADCAST_Sx(&ARow[3]);
      acc00 = MM_FMADD(a, b0, acc00);
      acc10 = MM_FMADD(a, b1, acc10);

      a = MM_BROADCAST_Sx(&ARow[lda1+3]);
      acc01 = MM_FMADD(a, b0, acc01);
      acc11 = MM_FMADD(a, b1, acc11);

      a = MM_BROADCAST_Sx(&ARow[lda2+3]);
      acc02 = MM_FMADD(a, b0, acc02);
      acc12 = MM_FMADD(a, b1, acc12);

      a = MM_BROADCAST_Sx(&ARow[lda3+3]);
      acc03 = MM_FMADD(a, b0, acc03);
      acc13 = MM_FMADD(a, b1, acc13);

      a = MM_BROADCAST_Sx(&ARow[lda4+3]);
      acc04 = MM_FMADD(a, b0, acc04);
      acc14 = MM_FMADD(a, b1, acc14);

      ARow += 4;
    }
    if (kRemSteps != 0) {
      fp_vector_t a, b0, b1;
      b0 = Bcol[0];
      b1 = Bcol[1];
      Bcol += B_WORDS_PER_ITER_MN2;

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

      if (kRemSteps != 1) {
        b0 = Bcol[0];
        b1 = Bcol[1];
        Bcol += B_WORDS_PER_ITER_MN2;

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

        if (kRemSteps != 2) {
          b0 = Bcol[0];
          b1 = Bcol[1];
          Bcol += B_WORDS_PER_ITER_MN2;

          a = MM_BROADCAST_Sx(&ARow[2]);
          acc00 = MM_FMADD(a, b0, acc00);
          acc10 = MM_FMADD(a, b1, acc10);

          a = MM_BROADCAST_Sx(&ARow[lda1+2]);
          acc01 = MM_FMADD(a, b0, acc01);
          acc11 = MM_FMADD(a, b1, acc11);

          a = MM_BROADCAST_Sx(&ARow[lda2+2]);
          acc02 = MM_FMADD(a, b0, acc02);
          acc12 = MM_FMADD(a, b1, acc12);

          a = MM_BROADCAST_Sx(&ARow[lda3+2]);
          acc03 = MM_FMADD(a, b0, acc03);
          acc13 = MM_FMADD(a, b1, acc13);

          a = MM_BROADCAST_Sx(&ARow[lda4+2]);
          acc04 = MM_FMADD(a, b0, acc04);
          acc14 = MM_FMADD(a, b1, acc14);
        }
      }
    }

    fp_vector_t  alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
    int_vector_t mask = pPrm->mask_n;
    if (pPrm->c_option == C_OPTION_UPDATE) {
      #define UPDATE_CCOL(ccol, acc0, acc1) \
      MM_STOREU_Px    (&((ccol)[SIMD_FACTOR*0]),       MM_FMADD((acc0), alpha_ps, MM_LOADU_Px    (&((ccol)[SIMD_FACTOR*0])))); \
      MM_MASKSTOREU_Px(&((ccol)[SIMD_FACTOR*1]), mask, MM_FMADD((acc1), alpha_ps, MM_MASKLOADU_Px(&((ccol)[SIMD_FACTOR*1]), mask)));

      UPDATE_CCOL(C, acc00, acc10); C += ldc;
      UPDATE_CCOL(C, acc01, acc11); C += ldc;
      UPDATE_CCOL(C, acc02, acc12); C += ldc;
      UPDATE_CCOL(C, acc03, acc13); C += ldc;
      UPDATE_CCOL(C, acc04, acc14); C += ldc;

      #undef UPDATE_CCOL
    } else if (pPrm->c_option == C_OPTION_REPLACE) {
      #define UPDATE_CCOL(ccol, acc0, acc1) \
      MM_STOREU_Px    (&((ccol)[SIMD_FACTOR*0]),       MM_MUL_Px((acc0), alpha_ps)); \
      MM_MASKSTOREU_Px(&((ccol)[SIMD_FACTOR*1]), mask, MM_MUL_Px((acc1), alpha_ps));

      UPDATE_CCOL(C, acc00, acc10); C += ldc;
      UPDATE_CCOL(C, acc01, acc11); C += ldc;
      UPDATE_CCOL(C, acc02, acc12); C += ldc;
      UPDATE_CCOL(C, acc03, acc13); C += ldc;
      UPDATE_CCOL(C, acc04, acc14); C += ldc;

      #undef UPDATE_CCOL
    } else { // C_OPTION_MULTIPLY
      fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
      #define UPDATE_CCOL(ccol, acc0, acc1) \
      MM_STOREU_Px    (&((ccol)[SIMD_FACTOR*0]),       MM_FMADD((acc0), alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px    (&((ccol)[SIMD_FACTOR*0]))))); \
      MM_MASKSTOREU_Px(&((ccol)[SIMD_FACTOR*1]), mask, MM_FMADD((acc1), alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(&((ccol)[SIMD_FACTOR*1]), mask))));

      UPDATE_CCOL(C, acc00, acc10); C += ldc;
      UPDATE_CCOL(C, acc01, acc11); C += ldc;
      UPDATE_CCOL(C, acc02, acc12); C += ldc;
      UPDATE_CCOL(C, acc03, acc13); C += ldc;
      UPDATE_CCOL(C, acc04, acc14); C += ldc;

      #undef UPDATE_CCOL
    }
  }

  // handle remaining rows of A
  m += A_WORDS_PER_ITER_MN2 - 1;
  for (; m > 0; A += lda1, --m) {
    const fp_vector_t* Bcol = pPrm->bb;
    fp_vector_t acc00 = MM_SETZERO_Px();
    fp_vector_t acc01 = MM_SETZERO_Px();
    fp_vector_t acc02 = MM_SETZERO_Px();
    fp_vector_t acc03 = MM_SETZERO_Px();

    fp_vector_t acc10 = MM_SETZERO_Px();
    fp_vector_t acc11 = MM_SETZERO_Px();
    fp_vector_t acc12 = MM_SETZERO_Px();
    fp_vector_t acc13 = MM_SETZERO_Px();

    _mm_prefetch((char*)(C),                         _MM_HINT_T0);
    _mm_prefetch((char*)(C)+sizeof(fp_vector_t)*2-1, _MM_HINT_T0);
    const scalar_t* ARow = A;
    for (int k = 0; k < kFullSteps; ++k) {
      fp_vector_t a;

      a = MM_BROADCAST_Sx(&ARow[0]);
      acc00 = MM_FMADD(a, Bcol[0], acc00);
      acc10 = MM_FMADD(a, Bcol[1], acc10);

      a = MM_BROADCAST_Sx(&ARow[1]);
      acc01 = MM_FMADD(a, Bcol[2], acc01);
      acc11 = MM_FMADD(a, Bcol[3], acc11);

      a = MM_BROADCAST_Sx(&ARow[2]);
      acc02 = MM_FMADD(a, Bcol[4], acc02);
      acc12 = MM_FMADD(a, Bcol[5], acc12);

      a = MM_BROADCAST_Sx(&ARow[3]);
      acc03 = MM_FMADD(a, Bcol[6], acc03);
      acc13 = MM_FMADD(a, Bcol[7], acc13);

      Bcol += 8;
      ARow += 4;
    }
    if (kRemSteps != 0) {
      fp_vector_t a;
      a = MM_BROADCAST_Sx(&ARow[0]);
      acc00 = MM_FMADD(a, Bcol[0], acc00);
      acc10 = MM_FMADD(a, Bcol[1], acc10);
      if (kRemSteps != 1) {
        a = MM_BROADCAST_Sx(&ARow[1]);
        acc01 = MM_FMADD(a, Bcol[2], acc01);
        acc11 = MM_FMADD(a, Bcol[3], acc11);
        if (kRemSteps != 2) {
          a = MM_BROADCAST_Sx(&ARow[2]);
          acc02 = MM_FMADD(a, Bcol[4], acc02);
          acc12 = MM_FMADD(a, Bcol[5], acc12);
        }
      }
    }
    acc00 = MM_ADD_Px(acc00, acc01);
    acc02 = MM_ADD_Px(acc02, acc03);
    acc10 = MM_ADD_Px(acc10, acc11);
    acc12 = MM_ADD_Px(acc12, acc13);

    acc00 = MM_ADD_Px(acc00, acc02);
    acc10 = MM_ADD_Px(acc10, acc12);

    fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
    int_vector_t mask = pPrm->mask_n;
    if (pPrm->c_option == C_OPTION_UPDATE) {
      MM_STOREU_Px    (&C[SIMD_FACTOR*0],       MM_FMADD(acc00, alpha_ps, MM_LOADU_Px    (&C[SIMD_FACTOR*0])));
      MM_MASKSTOREU_Px(&C[SIMD_FACTOR*1], mask, MM_FMADD(acc10, alpha_ps, MM_MASKLOADU_Px(&C[SIMD_FACTOR*1], mask)));
    } else if (pPrm->c_option == C_OPTION_REPLACE) {
      MM_STOREU_Px    (&C[SIMD_FACTOR*0],       MM_MUL_Px(acc00, alpha_ps));
      MM_MASKSTOREU_Px(&C[SIMD_FACTOR*1], mask, MM_MUL_Px(acc10, alpha_ps));
    } else { // C_OPTION_MULTIPLY
      fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
      MM_STOREU_Px    (&C[SIMD_FACTOR*0],       MM_FMADD(acc00, alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px    (&C[SIMD_FACTOR*0]))));
      MM_MASKSTOREU_Px(&C[SIMD_FACTOR*1], mask, MM_FMADD(acc10, alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(&C[SIMD_FACTOR*1], mask))));
    }
    C += ldc;
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
  int n_bIters, int nRows)
{
  fp_vector_t* dstCol = pPrm->bb;
  int ldbb = nRows*B_WORDS_PER_ITER;
  int_vector_t  mask_n = pPrm->mask_n;
  for (int r = 0; r < nRows; ++r) {
    const scalar_t *src = B;
    fp_vector_t* dst = dstCol;
    for (int c = 0; c < n_bIters; ++c) {
      for (int w = 0; w < B_WORDS_PER_ITER-1; ++w)
        dst[w] = MM_LOADU_Px(&src[w*SIMD_FACTOR]);
      if (c != pPrm->masked_b_it)
        dst[B_WORDS_PER_ITER-1] = MM_LOADU_Px(&src[(B_WORDS_PER_ITER-1)*SIMD_FACTOR]);
      else
        dst[B_WORDS_PER_ITER-1] = MM_MASKLOADU_Px(&src[(B_WORDS_PER_ITER-1)*SIMD_FACTOR], mask_n);
      src += SIMD_FACTOR*B_WORDS_PER_ITER;
      dst += ldbb;
    }
    B      += ldb;
    dstCol += B_WORDS_PER_ITER;
  }
}

static void CopyAndTransposeMn1WithMask(
  noncblas_sgemm_prm_t* pPrm,
  const scalar_t *B, int ldb,
  int nRows)
{
  int_vector_t mask = pPrm->mask_n;
  for (int r = 0; r < nRows; B += ldb, ++r)
    pPrm->bb[r] = MM_MASKLOADU_Px(B, mask);
}

static void CopyAndTransposeMn2WithMask(
  noncblas_sgemm_prm_t* pPrm,
  const scalar_t *B, int ldb,
  int nRows)
{
  int_vector_t mask = pPrm->mask_n;
  fp_vector_t* dst = pPrm->bb;
  for (int r = 0; r < nRows; B += ldb, dst += 2, ++r) {
    dst[0] = MM_LOADU_Px    (&B[SIMD_FACTOR*0]);
    dst[1] = MM_MASKLOADU_Px(&B[SIMD_FACTOR*1], mask);
  }
}


static int st_m_step = 0;
static int st_k_step = 0;
extern uint64_t dbg_tt;
#include <stdio.h>
#include <x86intrin.h>

// N>SIMD_FACTOR
// K>7
static void noncblas_sgemm_wide_n(
  int M, int N, int K,
  scalar_t alpha,
  const scalar_t *A, int lda,
  const scalar_t *B, int ldb,
  scalar_t beta,
  scalar_t *C, int ldc)
{
  unsigned nw = (unsigned)(N - 1) / SIMD_FACTOR + 1;
  int nwMj = ((nw - 1) / SIMD_ELEM_PEC_COL_MJ) * SIMD_ELEM_PEC_COL_MJ;
  int nwRem = nw - nwMj;
  int nwRemMj = nwRem / B_WORDS_PER_ITER;
  int nwRemMn = nwRem - nwRemMj*B_WORDS_PER_ITER;
  int nMj = nwMj * SIMD_FACTOR;

  const int K_STEP_NOM = ((unsigned)K_STEP/2)*2 + 1;
  const int K_STEP_MAX = ((unsigned)K_STEP_NOM/2)*3;
  int k_step = K > K_STEP_MAX ? K_STEP_NOM : K;

#ifdef USE_CONSTANT_M_STEP
  int M_STEP_NOM = M_STEP;
#else
  int M_STEP_NOM = MxN_BLOCK_SZ/(N*sizeof(scalar_t));
#endif
  M_STEP_NOM = ((unsigned)(M_STEP_NOM-1)/(A_WORDS_PER_ITER*2) + 1)*(A_WORDS_PER_ITER*2);
  const int M_STEP_MAX = (M_STEP_NOM/2)*3;
  int m_step = M > M_STEP_MAX ? M_STEP_NOM : M;



  const int bb_sz = SIMD_ELEM_PEC_COL_MJ*k_step;
  const int workBufSz = bb_sz;
  // I didn't find a standard portable way to allocate 32-byte aligned buffer
  // So I am doing it in hackish, but reliable way
  char* workBufAlloc = malloc((workBufSz+1)*sizeof(fp_vector_t));
  uintptr_t workBufAdj = (0-(uintptr_t)(workBufAlloc)) % sizeof(fp_vector_t);
  fp_vector_t* workBuf = (fp_vector_t*)(workBufAlloc+workBufAdj);

  noncblas_sgemm_prm_t prm;
  prm.bb  = workBuf;
  prm.lda = lda;
  prm.ldc = ldc;
  prm.alpha = alpha;
  prm.beta  = beta;
  memset(&prm.mask_n, -1, sizeof(prm.mask_n));
  int nwRemMj_masked_b_it = -1;
  unsigned nRem = (unsigned)N % SIMD_FACTOR;
  if (nRem > 0) { // mask off elements of rightmost SIMD word in B and C
    memset(&prm.mask_n, 0, sizeof(prm.mask_n));
    memset((char*)&prm.mask_n, -1, sizeof(*C)*nRem);
    if (nwRemMn == 0)
      nwRemMj_masked_b_it =  nwRemMj - 1;
  }

  //printf("nMj=%d, nwRemMn=%d, nwRemMj=%d remW_n=%d n_step=%d\n", nMj, nwRemMn, nwRemMj, remW_n, n_step);
  uint64_t tt = 0;

  for (int k = 0; k < K; k += k_step) {
    prm.c_option = C_OPTION_UPDATE;
    if (k==0 && prm.beta != 1.0f)
      prm.c_option = (prm.beta == 0) ? C_OPTION_REPLACE : C_OPTION_MULTIPLY;
    int delta_k = K - k;
    if (delta_k > k_step) {
      if (delta_k < K_STEP_MAX)
        k_step = ((unsigned)(delta_k-1)/4 + 1)*2 + 1;
      delta_k = k_step;
    }
    for (int m = 0; m < M; m += prm.M) {
      int delta_m = M - m;
      if (delta_m > m_step) {
        if (delta_m < M_STEP_MAX)
          m_step = ((unsigned)(delta_m-1)/(A_WORDS_PER_ITER*2) + 1)*A_WORDS_PER_ITER;
        delta_m = m_step;
      }
      prm.M = delta_m;

      prm.masked_b_it = -1;          // all words in use
      prm.A = &A[m*lda+k];

      scalar_t *Crow = &C[m*ldc];
      int n;
      for (n = 0; n < nMj; n += n_step) {
        // process full-width major rectangles
        uint64_t t0 = __rdtsc();
        CopyAndTransposeMj(&prm, &B[k*ldb + n], ldb, N_STEP_MULTIPLIER, delta_k);
        uint64_t t1 = __rdtsc();
        tt += t1 - t0;
        fma256_noncblas_sgemm_core_mj(&prm, &Crow[n], N_STEP_MULTIPLIER, delta_k);
      }
      if (nwRemMj > 0) {
        // process rightmost major rectangle, either full or partial
        prm.masked_b_it = nwRemMj_masked_b_it;
        CopyAndTransposeMjWithMask(&prm, &B[k*ldb + n], ldb, nwRemMj, delta_k);
        fma256_noncblas_sgemm_core_mj(&prm, &Crow[n], nwRemMj, delta_k);
        n += nwRemMj*B_WORDS_PER_ITER*SIMD_FACTOR;
      }
      if (nwRemMn > 0) {
        if (nwRemMn == 1) {
          CopyAndTransposeMn1WithMask(&prm, &B[k*ldb + n], ldb, delta_k);
          fma256_noncblas_sgemm_core_mn1(&prm, &Crow[n], delta_k);
        } else {
          CopyAndTransposeMn2WithMask(&prm, &B[k*ldb + n], ldb, delta_k);
          fma256_noncblas_sgemm_core_mn2(&prm, &Crow[n], delta_k);
        }
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

#include "avxnnn_noncblas_sgemm_smallK.c"

void func_name(
  int M, int N, int K,
  scalar_t alpha,
  const scalar_t *A, int lda,
  const scalar_t *B, int ldb,
  scalar_t beta,
  scalar_t *C, int ldc)
{
  if (M <= 0 || N <= 0 || K <= 0)
    return;
  if (K >= 8) {
    if (N > SIMD_FACTOR) {
      noncblas_sgemm_wide_n(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    } else if (N >= 1) {
      noncblas_sgemm_narrow_n(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
  } else {
    noncblas_sgemm_smallK(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  _mm256_zeroupper();
}


void tune_name(int m_step, int k_step) {
  st_m_step = m_step;
  st_k_step = k_step;
}