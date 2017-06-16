enum {
 SIMD_FACTOR          = sizeof(fp_vector_t)/sizeof(scalar_t),
 A_WORDS_PER_ITER     = 5,
 B_WORDS_PER_ITER     = 2,
 SIMD_ELEM_PEC_COL_MJ = B_WORDS_PER_ITER*N_STEP_MULTIPLIER,
 n_step               = SIMD_ELEM_PEC_COL_MJ*SIMD_FACTOR,
};

enum {
  C_OPTION_UPDATE, C_OPTION_REPLACE, C_OPTION_MULTIPLY
};

typedef struct {
  int           mDiv;
  int           mRem;
  int           lda;
  int           ldc;
  int           c_option;
  int           masked_b_it;
  scalar_t      alpha;
  scalar_t      beta;
  int_vector_t  mask_n;
  int_vector4_t mask_k;
  fp_vector_t*  bb; // [SIMD_ELEM_PEC_COL_MJ*k_step];
  fp_vector4_t* aa; // [(k_step/4)*m_step_max];
} noncblas_sgemm_prm_t;

// major core - inner loop processes 2 SIMD columns of B x 5 rows of A
static void fma256_noncblas_sgemm_core_mj(
 const noncblas_sgemm_prm_t* pPrm,
 scalar_t*                   C,
 int                         n_bIters, // 0 < n_bIters <= N_STEP_MULTIPLIER
 int                         kSteps)   // 1 < kSteps <= k_step/4
{
  const int       ldc      = pPrm->ldc;
  const int       ldaa     = A_WORDS_PER_ITER*kSteps*4;
  const int       ldbb     = B_WORDS_PER_ITER*kSteps*4;
  const scalar_t* A        = (const scalar_t*)pPrm->aa;
  const int       b_itLast = n_bIters - 1;
  #define PrefetchLines(x) \
    _mm_prefetch((char*)(x), _MM_HINT_T0); \
    _mm_prefetch((char*)(x)+sizeof(fp_vector_t)*B_WORDS_PER_ITER-1, _MM_HINT_T0);

  for (int m = pPrm->mDiv; m > 0; A += ldaa, C += ldc*A_WORDS_PER_ITER, --m) {
    scalar_t* Crow = C;
    for (int b_it = 0; b_it <= b_itLast; Crow += B_WORDS_PER_ITER*SIMD_FACTOR, ++b_it) {
      const fp_vector_t* Bcol = &pPrm->bb[ldbb*b_it];
      const scalar_t* ARow = A;

      fp_vector_t a;
      fp_vector_t b0 = Bcol[0];
      fp_vector_t b1 = Bcol[1];
      Bcol += B_WORDS_PER_ITER;

      scalar_t* CPrefetch = Crow;
      a = MM_BROADCAST_Sx(&ARow[4*0+0]);
      fp_vector_t acc00 = MM_MUL_Px(a, b0);
      fp_vector_t acc10 = MM_MUL_Px(a, b1);
      PrefetchLines(CPrefetch); CPrefetch += ldc;

      a = MM_BROADCAST_Sx(&ARow[4*1+0]);
      fp_vector_t acc01 = MM_MUL_Px(a, b0);
      fp_vector_t acc11 = MM_MUL_Px(a, b1);
      PrefetchLines(CPrefetch); CPrefetch += ldc;

      a = MM_BROADCAST_Sx(&ARow[4*2+0]);
      fp_vector_t acc02 = MM_MUL_Px(a, b0);
      fp_vector_t acc12 = MM_MUL_Px(a, b1);
      PrefetchLines(CPrefetch); CPrefetch += ldc;

      a = MM_BROADCAST_Sx(&ARow[4*3+0]);
      fp_vector_t acc03 = MM_MUL_Px(a, b0);
      fp_vector_t acc13 = MM_MUL_Px(a, b1);
      PrefetchLines(CPrefetch); CPrefetch += ldc;

      a = MM_BROADCAST_Sx(&ARow[4*4+0]);
      fp_vector_t acc04 = MM_MUL_Px(a, b0);
      fp_vector_t acc14 = MM_MUL_Px(a, b1);
      PrefetchLines(CPrefetch); CPrefetch += ldc;

      #define MADD_STEP5x2(a_offset) \
      b0 = Bcol[0];                               \
      b1 = Bcol[1];                               \
      Bcol += B_WORDS_PER_ITER;                   \
                                                  \
      a = MM_BROADCAST_Sx(&ARow[4*0+(a_offset)]); \
      acc00 = MM_FMADD(a, b0, acc00);             \
      acc10 = MM_FMADD(a, b1, acc10);             \
                                                  \
      a = MM_BROADCAST_Sx(&ARow[4*1+(a_offset)]); \
      acc01 = MM_FMADD(a, b0, acc01);             \
      acc11 = MM_FMADD(a, b1, acc11);             \
                                                  \
      a = MM_BROADCAST_Sx(&ARow[4*2+(a_offset)]); \
      acc02 = MM_FMADD(a, b0, acc02);             \
      acc12 = MM_FMADD(a, b1, acc12);             \
                                                  \
      a = MM_BROADCAST_Sx(&ARow[4*3+(a_offset)]); \
      acc03 = MM_FMADD(a, b0, acc03);             \
      acc13 = MM_FMADD(a, b1, acc13);             \
                                                  \
      a = MM_BROADCAST_Sx(&ARow[4*4+(a_offset)]); \
      acc04 = MM_FMADD(a, b0, acc04);             \
      acc14 = MM_FMADD(a, b1, acc14);

      MADD_STEP5x2(1)
      MADD_STEP5x2(2)
      MADD_STEP5x2(3)
      ARow += A_WORDS_PER_ITER*4;

      int k = kSteps-1;
      do {
        MADD_STEP5x2(0)
        MADD_STEP5x2(1)
        MADD_STEP5x2(2)
        MADD_STEP5x2(3)
        ARow += A_WORDS_PER_ITER*4;
      } while (--k);
      #undef MADD_STEP5x2

      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
      scalar_t* CCol = Crow;
      if (b_it != pPrm->masked_b_it) {
        if (pPrm->c_option == C_OPTION_UPDATE) {
          #define UPDATE_CCOL(ccol, acc0, acc1) \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*0]), MM_FMADD((acc0), alpha_ps, MM_LOADU_Px(&((ccol)[SIMD_FACTOR*0])))); \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*1]), MM_FMADD((acc1), alpha_ps, MM_LOADU_Px(&((ccol)[SIMD_FACTOR*1]))));

          UPDATE_CCOL(CCol, acc00, acc10); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11); CCol += ldc;
          UPDATE_CCOL(CCol, acc02, acc12); CCol += ldc;
          UPDATE_CCOL(CCol, acc03, acc13); CCol += ldc;
          UPDATE_CCOL(CCol, acc04, acc14);

          #undef UPDATE_CCOL
        } else if (pPrm->c_option == C_OPTION_REPLACE) {
          #define UPDATE_CCOL(ccol, acc0, acc1) \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*0]), MM_MUL_Px((acc0), alpha_ps)); \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*1]), MM_MUL_Px((acc1), alpha_ps));

          UPDATE_CCOL(CCol, acc00, acc10); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11); CCol += ldc;
          UPDATE_CCOL(CCol, acc02, acc12); CCol += ldc;
          UPDATE_CCOL(CCol, acc03, acc13); CCol += ldc;
          UPDATE_CCOL(CCol, acc04, acc14);

          #undef UPDATE_CCOL
        } else { // C_OPTION_MULTIPLY
          fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
          #define UPDATE_CCOL(ccol, acc0, acc1) \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*0]), MM_FMADD((acc0), alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px(&((ccol)[SIMD_FACTOR*0]))))); \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*1]), MM_FMADD((acc1), alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px(&((ccol)[SIMD_FACTOR*1])))));

          UPDATE_CCOL(CCol, acc00, acc10); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11); CCol += ldc;
          UPDATE_CCOL(CCol, acc02, acc12); CCol += ldc;
          UPDATE_CCOL(CCol, acc03, acc13); CCol += ldc;
          UPDATE_CCOL(CCol, acc04, acc14);

          #undef UPDATE_CCOL
        }
      } else {
        int_vector_t mask = pPrm->mask_n;
        if (pPrm->c_option == C_OPTION_UPDATE) {
          #define UPDATE_CCOL(ccol, acc0, acc1) \
          MM_STOREU_Px    (&((ccol)[SIMD_FACTOR*0]),       MM_FMADD((acc0), alpha_ps, MM_LOADU_Px    (&((ccol)[SIMD_FACTOR*0])))); \
          MM_MASKSTOREU_Px(&((ccol)[SIMD_FACTOR*1]), mask, MM_FMADD((acc1), alpha_ps, MM_MASKLOADU_Px(&((ccol)[SIMD_FACTOR*1]), mask)));

          UPDATE_CCOL(CCol, acc00, acc10); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11); CCol += ldc;
          UPDATE_CCOL(CCol, acc02, acc12); CCol += ldc;
          UPDATE_CCOL(CCol, acc03, acc13); CCol += ldc;
          UPDATE_CCOL(CCol, acc04, acc14);

          #undef UPDATE_CCOL
        } else if (pPrm->c_option == C_OPTION_REPLACE) {
          #define UPDATE_CCOL(ccol, acc0, acc1) \
          MM_STOREU_Px    (&((ccol)[SIMD_FACTOR*0]),       MM_MUL_Px((acc0), alpha_ps)); \
          MM_MASKSTOREU_Px(&((ccol)[SIMD_FACTOR*1]), mask, MM_MUL_Px((acc1), alpha_ps));

          UPDATE_CCOL(CCol, acc00, acc10); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11); CCol += ldc;
          UPDATE_CCOL(CCol, acc02, acc12); CCol += ldc;
          UPDATE_CCOL(CCol, acc03, acc13); CCol += ldc;
          UPDATE_CCOL(CCol, acc04, acc14);

          #undef UPDATE_CCOL
        } else { // C_OPTION_MULTIPLY
          fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
          #define UPDATE_CCOL(ccol, acc0, acc1) \
          MM_STOREU_Px    (&((ccol)[SIMD_FACTOR*0]),       MM_FMADD((acc0), alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px    (&((ccol)[SIMD_FACTOR*0]))))); \
          MM_MASKSTOREU_Px(&((ccol)[SIMD_FACTOR*1]), mask, MM_FMADD((acc1), alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(&((ccol)[SIMD_FACTOR*1]), mask))));

          UPDATE_CCOL(CCol, acc00, acc10); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11); CCol += ldc;
          UPDATE_CCOL(CCol, acc02, acc12); CCol += ldc;
          UPDATE_CCOL(CCol, acc03, acc13); CCol += ldc;
          UPDATE_CCOL(CCol, acc04, acc14);

          #undef UPDATE_CCOL
        }
      }
    }
  }

  // handle remaining rows of a - non-interleaved, padded to multiple of 4
  for (int m = pPrm->mRem; m > 0; A += kSteps*4, C += ldc, --m) {
    scalar_t* Crow = C;
    for (int b_it = 0; b_it <= b_itLast; Crow += B_WORDS_PER_ITER*SIMD_FACTOR, ++b_it) {
      const fp_vector_t* Bcol = &pPrm->bb[ldbb*b_it];
      const scalar_t*    ARow = A;

      fp_vector_t a;

      a = MM_BROADCAST_Sx(&ARow[0]);
      fp_vector_t acc00 = MM_MUL_Px(a, Bcol[0]);
      fp_vector_t acc10 = MM_MUL_Px(a, Bcol[1]);
      Bcol += B_WORDS_PER_ITER;

      a = MM_BROADCAST_Sx(&ARow[1]);
      fp_vector_t acc01 = MM_MUL_Px(a, Bcol[0]);
      fp_vector_t acc11 = MM_MUL_Px(a, Bcol[1]);;
      Bcol += B_WORDS_PER_ITER;

      a = MM_BROADCAST_Sx(&ARow[2]);
      fp_vector_t acc02 = MM_MUL_Px(a, Bcol[0]);
      fp_vector_t acc12 = MM_MUL_Px(a, Bcol[1]);
      Bcol += B_WORDS_PER_ITER;

      a = MM_BROADCAST_Sx(&ARow[3]);
      fp_vector_t acc03 = MM_MUL_Px(a, Bcol[0]);
      fp_vector_t acc13 = MM_MUL_Px(a, Bcol[1]);
      Bcol += B_WORDS_PER_ITER;
      ARow += 4;

      PrefetchLines(Crow);
      int k = kSteps - 1;
      do {
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
      } while (--k);

      acc00 = MM_ADD_Px(acc00, acc01);
      acc02 = MM_ADD_Px(acc02, acc03);

      acc10 = MM_ADD_Px(acc10, acc11);
      acc12 = MM_ADD_Px(acc12, acc13);

      acc00 = MM_ADD_Px(acc00, acc02);
      acc10 = MM_ADD_Px(acc10, acc12);

      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
      if (b_it != pPrm->masked_b_it) {
        if (pPrm->c_option == C_OPTION_UPDATE) {
          MM_STOREU_Px(&Crow[SIMD_FACTOR*0], MM_FMADD(acc00, alpha_ps, MM_LOADU_Px(&Crow[SIMD_FACTOR*0])));
          MM_STOREU_Px(&Crow[SIMD_FACTOR*1], MM_FMADD(acc10, alpha_ps, MM_LOADU_Px(&Crow[SIMD_FACTOR*1])));
        } else if (pPrm->c_option == C_OPTION_REPLACE) {
          MM_STOREU_Px(&Crow[SIMD_FACTOR*0], MM_MUL_Px(acc00, alpha_ps));
          MM_STOREU_Px(&Crow[SIMD_FACTOR*1], MM_MUL_Px(acc10, alpha_ps));
        } else { // C_OPTION_MULTIPLY
          fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
          MM_STOREU_Px(&Crow[SIMD_FACTOR*0], MM_FMADD(acc00, alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px(&Crow[SIMD_FACTOR*0]))));
          MM_STOREU_Px(&Crow[SIMD_FACTOR*1], MM_FMADD(acc10, alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px(&Crow[SIMD_FACTOR*1]))));
        }
      } else {
        int_vector_t mask = pPrm->mask_n;
        if (pPrm->c_option == C_OPTION_UPDATE) {
          MM_STOREU_Px    (&Crow[SIMD_FACTOR*0],       MM_FMADD(acc00, alpha_ps, MM_LOADU_Px    (&Crow[SIMD_FACTOR*0])));
          MM_MASKSTOREU_Px(&Crow[SIMD_FACTOR*1], mask, MM_FMADD(acc10, alpha_ps, MM_MASKLOADU_Px(&Crow[SIMD_FACTOR*1], mask)));
        } else if (pPrm->c_option == C_OPTION_REPLACE) {
          MM_STOREU_Px    (&Crow[SIMD_FACTOR*0],       MM_MUL_Px(acc00, alpha_ps));
          MM_MASKSTOREU_Px(&Crow[SIMD_FACTOR*1], mask, MM_MUL_Px(acc10, alpha_ps));
        } else { // C_OPTION_MULTIPLY
          fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
          MM_STOREU_Px    (&Crow[SIMD_FACTOR*0],       MM_FMADD(acc00, alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px    (&Crow[SIMD_FACTOR*0]))));
          MM_MASKSTOREU_Px(&Crow[SIMD_FACTOR*1], mask, MM_FMADD(acc10, alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(&Crow[SIMD_FACTOR*1], mask))));
        }
      }
    }
  }
  #undef PrefetchLines
}

// minor core - inner loop processes 1 SIMD columns of B x 5 rows of A
static void fma256_noncblas_sgemm_core_mn(
 const noncblas_sgemm_prm_t* pPrm,
 scalar_t*                   C,
 int                         kSteps)   // 1 < kSteps <= k_step/4
{
  const int ldc = pPrm->ldc;
  const scalar_t* ARow = (const scalar_t*)pPrm->aa;
  for (int m = pPrm->mDiv; m > 0; --m) {
    const fp_vector_t* Bcol = pPrm->bb;
    fp_vector_t b;

    b = Bcol[0];
    fp_vector_t acc00 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[4*0+0]), b);
    fp_vector_t acc01 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[4*1+0]), b);
    fp_vector_t acc02 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[4*2+0]), b);
    fp_vector_t acc03 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[4*3+0]), b);
    fp_vector_t acc04 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[4*4+0]), b);

    b = Bcol[1];
    fp_vector_t acc10 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[4*0+1]), b);
    fp_vector_t acc11 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[4*1+1]), b);
    fp_vector_t acc12 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[4*2+1]), b);
    fp_vector_t acc13 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[4*3+1]), b);
    fp_vector_t acc14 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[4*4+1]), b);

    #define MADD_STEP5x1(acc0, acc1, acc2, acc3, acc4, a_offset) \
      b = Bcol[(a_offset)]; \
      acc0 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4*0+(a_offset)]), b, acc0); \
      acc1 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4*1+(a_offset)]), b, acc1); \
      acc2 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4*2+(a_offset)]), b, acc2); \
      acc3 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4*3+(a_offset)]), b, acc3); \
      acc4 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4*4+(a_offset)]), b, acc4);


    MADD_STEP5x1(acc01, acc01, acc02, acc03, acc04, 2)
    MADD_STEP5x1(acc11, acc11, acc12, acc13, acc14, 3)
    Bcol += 4;
    ARow += 4 * A_WORDS_PER_ITER;

    scalar_t* CPrefetch = C;
    _mm_prefetch((char*)(CPrefetch), _MM_HINT_T0); CPrefetch += ldc;
    _mm_prefetch((char*)(CPrefetch), _MM_HINT_T0); CPrefetch += ldc;
    _mm_prefetch((char*)(CPrefetch), _MM_HINT_T0); CPrefetch += ldc;
    _mm_prefetch((char*)(CPrefetch), _MM_HINT_T0); CPrefetch += ldc;
    _mm_prefetch((char*)(CPrefetch), _MM_HINT_T0); CPrefetch += ldc;

    int k = kSteps - 1;
    do {
      MADD_STEP5x1(acc01, acc01, acc02, acc03, acc04, 0)
      MADD_STEP5x1(acc11, acc11, acc12, acc13, acc14, 1)
      MADD_STEP5x1(acc01, acc01, acc02, acc03, acc04, 2)
      MADD_STEP5x1(acc11, acc11, acc12, acc13, acc14, 3)
      Bcol += 4;
      ARow += 4 * A_WORDS_PER_ITER;
    } while (--k);
    #undef MADD_STEP5x1

    acc00 = MM_ADD_Px(acc00, acc10);
    acc01 = MM_ADD_Px(acc01, acc11);
    acc02 = MM_ADD_Px(acc02, acc12);
    acc03 = MM_ADD_Px(acc03, acc13);
    acc04 = MM_ADD_Px(acc04, acc14);

    fp_vector_t  alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
    int_vector_t mask = pPrm->mask_n;

    if (pPrm->c_option == C_OPTION_UPDATE) {
      MM_MASKSTOREU_Px(C, mask, MM_FMADD(acc00, alpha_ps, MM_MASKLOADU_Px(C, mask))); C += ldc;
      MM_MASKSTOREU_Px(C, mask, MM_FMADD(acc01, alpha_ps, MM_MASKLOADU_Px(C, mask))); C += ldc;
      MM_MASKSTOREU_Px(C, mask, MM_FMADD(acc02, alpha_ps, MM_MASKLOADU_Px(C, mask))); C += ldc;
      MM_MASKSTOREU_Px(C, mask, MM_FMADD(acc03, alpha_ps, MM_MASKLOADU_Px(C, mask))); C += ldc;
      MM_MASKSTOREU_Px(C, mask, MM_FMADD(acc04, alpha_ps, MM_MASKLOADU_Px(C, mask))); C += ldc;
    } else if (pPrm->c_option == C_OPTION_REPLACE) {
      MM_MASKSTOREU_Px(C, mask, MM_MUL_Px(acc00, alpha_ps)); C += ldc;
      MM_MASKSTOREU_Px(C, mask, MM_MUL_Px(acc01, alpha_ps)); C += ldc;
      MM_MASKSTOREU_Px(C, mask, MM_MUL_Px(acc02, alpha_ps)); C += ldc;
      MM_MASKSTOREU_Px(C, mask, MM_MUL_Px(acc03, alpha_ps)); C += ldc;
      MM_MASKSTOREU_Px(C, mask, MM_MUL_Px(acc04, alpha_ps)); C += ldc;
    } else { // C_OPTION_MULTIPLY
      fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
      MM_MASKSTOREU_Px(C, mask, MM_FMADD(acc00, alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(C, mask)))); C += ldc;
      MM_MASKSTOREU_Px(C, mask, MM_FMADD(acc01, alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(C, mask)))); C += ldc;
      MM_MASKSTOREU_Px(C, mask, MM_FMADD(acc02, alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(C, mask)))); C += ldc;
      MM_MASKSTOREU_Px(C, mask, MM_FMADD(acc03, alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(C, mask)))); C += ldc;
      MM_MASKSTOREU_Px(C, mask, MM_FMADD(acc04, alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(C, mask)))); C += ldc;
    }
  }

  // handle remaining rows of a - non-interleaved
  for (int m = pPrm->mRem; m > 0; --m) {
    const fp_vector_t* Bcol = pPrm->bb;
    fp_vector_t acc00 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[0]), Bcol[0]);
    fp_vector_t acc01 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[1]), Bcol[1]);
    fp_vector_t acc02 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[2]), Bcol[2]);
    fp_vector_t acc03 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[3]), Bcol[3]);
    Bcol += 4;
    ARow += 4;

    _mm_prefetch((char*)(C), _MM_HINT_T0);

    int k = kSteps - 1;
    do {
      acc00 = MM_FMADD(MM_BROADCAST_Sx(&ARow[0]), Bcol[0], acc00);
      acc01 = MM_FMADD(MM_BROADCAST_Sx(&ARow[1]), Bcol[1], acc01);
      acc02 = MM_FMADD(MM_BROADCAST_Sx(&ARow[2]), Bcol[2], acc02);
      acc03 = MM_FMADD(MM_BROADCAST_Sx(&ARow[3]), Bcol[3], acc03);
      Bcol += 4;
      ARow += 4;
    } while (--k);

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


static void CopyAndTransposeMj(
  noncblas_sgemm_prm_t* pPrm,
  const scalar_t*       B, int ldb,
  int                   n_bIters,
  int                   nRows)
{
  fp_vector_t* dstCol = pPrm->bb;
  int ldbb = ((unsigned)(nRows+3)/4)*4*B_WORDS_PER_ITER;
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
  const scalar_t* B, int ldb,
  int             n_bIters,
  int             nRows)
{
  fp_vector_t* dstCol = pPrm->bb;
  int ldbb = ((unsigned)(nRows+3)/4)*4*B_WORDS_PER_ITER;
  int_vector_t mask = pPrm->mask_n;
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
  int_vector_t mask = pPrm->mask_n;
  for (int r = 0; r < nRows; B += ldb, ++r)
    pPrm->bb[r] = MM_MASKLOADU_Px(B, mask);
}

static void CopyAndInterleaveA(noncblas_sgemm_prm_t* pPrm, const scalar_t *A, int n_cols)
{
  int lda1 = pPrm->lda;
  int lda2 = lda1 + lda1;
  int lda3 = lda2 + lda1;
  int lda4 = lda3 + lda1;
  int lda5 = lda4 + lda1;
  fp_vector4_t* dst = pPrm->aa;
  // the bulk is interleaved
  for (int m = pPrm->mDiv; m > 0; A += lda5, --m) {
    const scalar_t *src = A;
    int k = (unsigned)n_cols / 4;
    do {
      fp_vector4_t a0 = MM_LOADU4_Px(&src[0]);
      fp_vector4_t a1 = MM_LOADU4_Px(&src[lda1]);
      fp_vector4_t a2 = MM_LOADU4_Px(&src[lda2]);
      fp_vector4_t a3 = MM_LOADU4_Px(&src[lda3]);
      fp_vector4_t a4 = MM_LOADU4_Px(&src[lda4]);
      src += 4;
      dst[0] = a0;
      dst[1] = a1;
      dst[2] = a2;
      dst[3] = a3;
      dst[4] = a4;
      dst += A_WORDS_PER_ITER;
    } while (--k);
    if ((unsigned)n_cols % 4 != 0) {
      int_vector4_t mask = pPrm->mask_k;
      fp_vector4_t a0 = MM_MASKLOADU4_Px(&src[0],    mask);
      fp_vector4_t a1 = MM_MASKLOADU4_Px(&src[lda1], mask);
      fp_vector4_t a2 = MM_MASKLOADU4_Px(&src[lda2], mask);
      fp_vector4_t a3 = MM_MASKLOADU4_Px(&src[lda3], mask);
      fp_vector4_t a4 = MM_MASKLOADU4_Px(&src[lda4], mask);
      dst[0] = a0;
      dst[1] = a1;
      dst[2] = a2;
      dst[3] = a3;
      dst[4] = a4;
      dst += A_WORDS_PER_ITER;
    }
  }

  // remaining rows not interleaved
  for (int m = pPrm->mRem; m > 0; A += lda1, --m) {
    const scalar_t *src = A;
    int k = (unsigned)n_cols / 4;
    do {
      dst[0] = MM_LOADU4_Px(&src[0]);
      src += 4;
      dst += 1;
    } while (--k);
    if ((unsigned)n_cols % 4 != 0) {
      dst[0] = MM_MASKLOADU4_Px(&src[0], pPrm->mask_k);
      dst += 1;
    }
  }
}

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
  int nMj       = (unsigned)N / n_step;
  unsigned nRem = (unsigned)N % n_step;
  int nwRem   = (nRem+SIMD_FACTOR-1) / SIMD_FACTOR;
  int nwRemMj = (unsigned)nwRem / B_WORDS_PER_ITER;
  int nwRemMn = (unsigned)nwRem % B_WORDS_PER_ITER;

  int m_step_nom = M;
  if (m_step_nom > (M_STEP/2)*3) {
    int m_Nsteps = (M-1)/M_STEP + 1;
    m_step_nom = ((M-1)/(m_Nsteps*A_WORDS_PER_ITER) + 1) * A_WORDS_PER_ITER;
  }

  // calculate k_step
  const int L1_BLOCK_N = L1_BLOCK_SZ/sizeof(scalar_t);
  const int L2_BLOCK_N = L2_BLOCK_SZ/sizeof(scalar_t);
  const int Neff = ((unsigned)(N-1) / n_step + 1) * n_step;
  const int C_N  = m_step_nom*Neff;
  // first try to fit everything into L1
  int k_step = (L1_BLOCK_N - C_N)/(m_step_nom+Neff);
  if (k_step < K_STEP_MIN) {
    // try to fit everything into L2
    int k_step_l2 = (L2_BLOCK_N - C_N)/(m_step_nom+Neff);
    // try to fit aa, bb and active area of C in L1
    int k_step_l1 = (L1_BLOCK_N - m_step_nom*n_step)/(m_step_nom+n_step);
    if (k_step_l2 >= K_STEP_MIN) {
      k_step = k_step_l2;
      if (k_step_l1 >= K_STEP_MIN) {
        if (k_step_l1 < k_step) {
          k_step = k_step_l1;
        }
      }
    } else if (k_step_l1 >= K_STEP_MIN) {
      k_step = k_step_l1;
    } else {
      // fit bb and active areas of aa and C in L1
      k_step = (L1_BLOCK_N - A_WORDS_PER_ITER*n_step)/(A_WORDS_PER_ITER+n_step);
    }
  }
  if (k_step < K) {
    int k_Nsteps = (K-1)/k_step + 1;
    k_step = ((K-1)/(k_Nsteps*4) + 1) * 4;
  } else {
    k_step = K;
  }

  static int uu = 1;
  if (uu) {
    printf("k_step=%d m_step=%d\n", k_step, m_step_nom);
    uu = 0;
  }

  const int k_step_ex = ((unsigned)(k_step-1)/4 + 1)*4;
  const int bb_sz = SIMD_ELEM_PEC_COL_MJ*k_step_ex;
  const int aa_sz = (m_step_nom*k_step_ex-1)/SIMD_FACTOR + 1;
  const int workBufSz = aa_sz + bb_sz;
  // I didn't find a standard portable way to allocate 32-byte aligned buffer
  // So I am doing it in hackish, but reliable way
  char* workBufAlloc = malloc((workBufSz+1)*sizeof(fp_vector_t));
  uintptr_t workBufAdj = (0-(uintptr_t)(workBufAlloc)) % sizeof(fp_vector_t);
  fp_vector_t* workBuf = (fp_vector_t*)(workBufAlloc+workBufAdj);

  noncblas_sgemm_prm_t prm;
  prm.aa = (fp_vector4_t*)(workBuf+0);
  prm.bb = (fp_vector_t*) (workBuf+aa_sz);
  prm.lda = lda;
  prm.ldc = ldc;
  prm.alpha = alpha;
  prm.beta  = beta;

  memset(&prm.mask_n, -1, sizeof(prm.mask_n));
  int nwRemMj_masked_b_it = -1;
  nRem %= SIMD_FACTOR;
  if (nRem > 0) { // mask off elements of rightmost SIMD word in B and C
    memset(&prm.mask_n, 0, sizeof(prm.mask_n));
    memset((char*)&prm.mask_n, -1, sizeof(*C)*nRem);
    if (nwRemMn == 0)
      nwRemMj_masked_b_it =  nwRemMj - 1;
  }

  memset(&prm.mask_k, -1, sizeof(prm.mask_k));
  unsigned kRem = (unsigned)K % 4;
  if (kRem > 0) { // mask off elements of rightmost SIMD word in A
    memset(&prm.mask_k, 0, sizeof(prm.mask_k));
    memset((char*)&prm.mask_k, -1, sizeof(*C)*kRem);
  }

  //printf("nMj=%d, nwRemMn=%d, nwRemMj=%d remW_n=%d n_step=%d\n", nMj, nwRemMn, nwRemMj, remW_n, n_step);
  uint64_t tt = 0;

  for (int k = 0; k < K; k += k_step) {
    prm.c_option = C_OPTION_UPDATE;
    if (k==0 && prm.beta != 1.0f)
      prm.c_option = (prm.beta == 0) ? C_OPTION_REPLACE : C_OPTION_MULTIPLY;
    int delta_k = K - k;
    if (delta_k > k_step) {
      if ((delta_k-k_step)*2 < k_step)
        k_step = ((unsigned)(delta_k-1)/(4*2) + 1)*4;
      delta_k = k_step;
    }
    const int kSteps = (unsigned)(delta_k-1)/4 + 1;

    int m_step = m_step_nom;
    for (int m = 0; m < M; m += m_step) {
      int delta_m = M - m;
      if (delta_m > m_step) {
        if ((delta_m - m_step)*2 < m_step)
          m_step = ((unsigned)(delta_m-1)/(A_WORDS_PER_ITER*2) + 1)*A_WORDS_PER_ITER;
        delta_m = m_step;
      }

      prm.mDiv = delta_m / A_WORDS_PER_ITER;
      prm.mRem = delta_m - prm.mDiv*A_WORDS_PER_ITER;

      prm.masked_b_it = -1;          // all words in use
      CopyAndInterleaveA(&prm, &A[m*lda+k], delta_k);

      scalar_t *Crow = &C[m*ldc];
      const scalar_t *Brow = &B[k*ldb];
      for (int ni = 0; ni < nMj; ++ni) {
        // process full-width major rectangles
        int n = ni * n_step;
        uint64_t t0 = __rdtsc();
        CopyAndTransposeMj(&prm, &Brow[n], ldb, N_STEP_MULTIPLIER, delta_k);
        uint64_t t1 = __rdtsc();
        tt += t1 - t0;
        fma256_noncblas_sgemm_core_mj(&prm, &Crow[n], N_STEP_MULTIPLIER, kSteps);
      }
      if (nwRemMj > 0) {
        prm.masked_b_it = nwRemMj_masked_b_it;
        int n = nMj * n_step;
        CopyAndTransposeMjWithMask(&prm, &Brow[n], ldb, nwRemMj, delta_k);
        fma256_noncblas_sgemm_core_mj(&prm, &Crow[n], nwRemMj, kSteps);
      }
      if (nwRemMn > 0) {
        int n = nMj * n_step + nwRemMj*B_WORDS_PER_ITER*SIMD_FACTOR;
        CopyAndTransposeMnWithMask(&prm, &Brow[n], ldb, delta_k);
        fma256_noncblas_sgemm_core_mn(&prm, &Crow[n], kSteps);
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
