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
  const scalar_t *A;
  const scalar_t *B;
  scalar_t       *C;
  unsigned      M, N, K;
  int           lda, ldb, ldc;
  int           mDiv;
  int           mRem;
  int           c_option;
  int           masked_b_it;
  int           hasMinor;
  scalar_t      alpha;
  scalar_t      beta;
  int_vector_t  mask_n;
  int_vector4_t mask_k;
  fp_vector_t*  bb; // [SIMD_ELEM_PEC_COL_MJ*k_step];
  fp_vector4_t* aa; // [(k_step/4)*m_step_max];
} noncblas_sgemm_prm_t;

// major core - inner loop processes 2 SIMD columns of B x 5 rows of A
static void fma256_noncblas_sgemm_core(
 const noncblas_sgemm_prm_t* pPrm,
 scalar_t*                   C,
 int                         n_bIters, // 0 < n_bIters <= N_STEP_MULTIPLIER
 int                         kSteps)   // 1 < kSteps <= k_step/4
{
  const int       ldc      = pPrm->ldc;
  const int       ldaa     = A_WORDS_PER_ITER*kSteps*4;
  const scalar_t* A        = (const scalar_t*)pPrm->aa;
  #define PrefetchLines(x) \
    _mm_prefetch((char*)(x), _MM_HINT_T0); \
    _mm_prefetch((char*)(x)+sizeof(fp_vector_t)*B_WORDS_PER_ITER-1, _MM_HINT_T0);

  for (int m = pPrm->mDiv; m > 0; A += ldaa, C += ldc*A_WORDS_PER_ITER, --m) {
    scalar_t* Crow = C;
    const fp_vector_t* Bcol = pPrm->bb;
    for (int b_it = 0; b_it < n_bIters; Crow += B_WORDS_PER_ITER*SIMD_FACTOR, ++b_it) {
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

    if (pPrm->hasMinor) {
      // handle minor column of B and C - a single SIMD column
      // An inner loop processes 1 SIMD columns of B x 5 rows of A
      const scalar_t* ARow = A;
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


      MADD_STEP5x1(acc00, acc01, acc02, acc03, acc04, 2)
      MADD_STEP5x1(acc10, acc11, acc12, acc13, acc14, 3)
      Bcol += 4;
      ARow += 4 * A_WORDS_PER_ITER;

      scalar_t* CPrefetch = Crow;
      _mm_prefetch((char*)(CPrefetch), _MM_HINT_T0); CPrefetch += ldc;
      _mm_prefetch((char*)(CPrefetch), _MM_HINT_T0); CPrefetch += ldc;
      _mm_prefetch((char*)(CPrefetch), _MM_HINT_T0); CPrefetch += ldc;
      _mm_prefetch((char*)(CPrefetch), _MM_HINT_T0); CPrefetch += ldc;
      _mm_prefetch((char*)(CPrefetch), _MM_HINT_T0); CPrefetch += ldc;

      int k = kSteps - 1;
      do {
        MADD_STEP5x1(acc00, acc01, acc02, acc03, acc04, 0)
        MADD_STEP5x1(acc10, acc11, acc12, acc13, acc14, 1)
        MADD_STEP5x1(acc00, acc01, acc02, acc03, acc04, 2)
        MADD_STEP5x1(acc10, acc11, acc12, acc13, acc14, 3)
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
      scalar_t* CCol = Crow;

      if (pPrm->c_option == C_OPTION_UPDATE) {
        MM_MASKSTOREU_Px(CCol, mask, MM_FMADD(acc00, alpha_ps, MM_MASKLOADU_Px(CCol, mask))); CCol += ldc;
        MM_MASKSTOREU_Px(CCol, mask, MM_FMADD(acc01, alpha_ps, MM_MASKLOADU_Px(CCol, mask))); CCol += ldc;
        MM_MASKSTOREU_Px(CCol, mask, MM_FMADD(acc02, alpha_ps, MM_MASKLOADU_Px(CCol, mask))); CCol += ldc;
        MM_MASKSTOREU_Px(CCol, mask, MM_FMADD(acc03, alpha_ps, MM_MASKLOADU_Px(CCol, mask))); CCol += ldc;
        MM_MASKSTOREU_Px(CCol, mask, MM_FMADD(acc04, alpha_ps, MM_MASKLOADU_Px(CCol, mask))); CCol += ldc;
      } else if (pPrm->c_option == C_OPTION_REPLACE) {
        MM_MASKSTOREU_Px(CCol, mask, MM_MUL_Px(acc00, alpha_ps)); CCol += ldc;
        MM_MASKSTOREU_Px(CCol, mask, MM_MUL_Px(acc01, alpha_ps)); CCol += ldc;
        MM_MASKSTOREU_Px(CCol, mask, MM_MUL_Px(acc02, alpha_ps)); CCol += ldc;
        MM_MASKSTOREU_Px(CCol, mask, MM_MUL_Px(acc03, alpha_ps)); CCol += ldc;
        MM_MASKSTOREU_Px(CCol, mask, MM_MUL_Px(acc04, alpha_ps)); CCol += ldc;
      } else { // C_OPTION_MULTIPLY
        fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
        MM_MASKSTOREU_Px(CCol, mask, MM_FMADD(acc00, alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(CCol, mask)))); CCol += ldc;
        MM_MASKSTOREU_Px(CCol, mask, MM_FMADD(acc01, alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(CCol, mask)))); CCol += ldc;
        MM_MASKSTOREU_Px(CCol, mask, MM_FMADD(acc02, alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(CCol, mask)))); CCol += ldc;
        MM_MASKSTOREU_Px(CCol, mask, MM_FMADD(acc03, alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(CCol, mask)))); CCol += ldc;
        MM_MASKSTOREU_Px(CCol, mask, MM_FMADD(acc04, alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(CCol, mask)))); CCol += ldc;
      }
    }
  }

  if (pPrm->mRem == 0)
    return;

  switch (pPrm->mRem) {
    case 4:
    {
    // process 4 rows of A
    const int a_words_per_iter = 4;
    scalar_t* Crow = C;
    const fp_vector_t* Bcol = pPrm->bb;
    for (int b_it = 0; b_it < n_bIters; Crow += B_WORDS_PER_ITER*SIMD_FACTOR, ++b_it) {
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

      #define MADD_STEP4x2(a_offset) \
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
      acc13 = MM_FMADD(a, b1, acc13);

      MADD_STEP4x2(1)
      MADD_STEP4x2(2)
      MADD_STEP4x2(3)
      ARow += a_words_per_iter*4;

      int k = kSteps-1;
      do {
        MADD_STEP4x2(0)
        MADD_STEP4x2(1)
        MADD_STEP4x2(2)
        MADD_STEP4x2(3)
        ARow += a_words_per_iter*4;
      } while (--k);
      #undef MADD_STEP4x2

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
          UPDATE_CCOL(CCol, acc03, acc13);

          #undef UPDATE_CCOL
        } else if (pPrm->c_option == C_OPTION_REPLACE) {
          #define UPDATE_CCOL(ccol, acc0, acc1) \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*0]), MM_MUL_Px((acc0), alpha_ps)); \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*1]), MM_MUL_Px((acc1), alpha_ps));

          UPDATE_CCOL(CCol, acc00, acc10); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11); CCol += ldc;
          UPDATE_CCOL(CCol, acc02, acc12); CCol += ldc;
          UPDATE_CCOL(CCol, acc03, acc13);

          #undef UPDATE_CCOL
        } else { // C_OPTION_MULTIPLY
          fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
          #define UPDATE_CCOL(ccol, acc0, acc1) \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*0]), MM_FMADD((acc0), alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px(&((ccol)[SIMD_FACTOR*0]))))); \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*1]), MM_FMADD((acc1), alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px(&((ccol)[SIMD_FACTOR*1])))));

          UPDATE_CCOL(CCol, acc00, acc10); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11); CCol += ldc;
          UPDATE_CCOL(CCol, acc02, acc12); CCol += ldc;
          UPDATE_CCOL(CCol, acc03, acc13);

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
          UPDATE_CCOL(CCol, acc03, acc13);


          #undef UPDATE_CCOL
        } else if (pPrm->c_option == C_OPTION_REPLACE) {
          #define UPDATE_CCOL(ccol, acc0, acc1) \
          MM_STOREU_Px    (&((ccol)[SIMD_FACTOR*0]),       MM_MUL_Px((acc0), alpha_ps)); \
          MM_MASKSTOREU_Px(&((ccol)[SIMD_FACTOR*1]), mask, MM_MUL_Px((acc1), alpha_ps));

          UPDATE_CCOL(CCol, acc00, acc10); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11); CCol += ldc;
          UPDATE_CCOL(CCol, acc02, acc12); CCol += ldc;
          UPDATE_CCOL(CCol, acc03, acc13);

          #undef UPDATE_CCOL
        } else { // C_OPTION_MULTIPLY
          fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
          #define UPDATE_CCOL(ccol, acc0, acc1) \
          MM_STOREU_Px    (&((ccol)[SIMD_FACTOR*0]),       MM_FMADD((acc0), alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px    (&((ccol)[SIMD_FACTOR*0]))))); \
          MM_MASKSTOREU_Px(&((ccol)[SIMD_FACTOR*1]), mask, MM_FMADD((acc1), alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(&((ccol)[SIMD_FACTOR*1]), mask))));

          UPDATE_CCOL(CCol, acc00, acc10); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11); CCol += ldc;
          UPDATE_CCOL(CCol, acc02, acc12); CCol += ldc;
          UPDATE_CCOL(CCol, acc03, acc13);

          #undef UPDATE_CCOL
        }
      }
    }

    if (pPrm->hasMinor) {
      // multiply of 4 rows of A by one SIMD column of B
      const scalar_t* ARow = A;
      fp_vector_t b;

      b = Bcol[0];
      fp_vector_t acc00 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[4*0+0]), b);
      fp_vector_t acc01 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[4*1+0]), b);
      fp_vector_t acc02 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[4*2+0]), b);
      fp_vector_t acc03 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[4*3+0]), b);

      b = Bcol[1];
      fp_vector_t acc10 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[4*0+1]), b);
      fp_vector_t acc11 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[4*1+1]), b);
      fp_vector_t acc12 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[4*2+1]), b);
      fp_vector_t acc13 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[4*3+1]), b);

      #define MADD_STEP4x1(acc0, acc1, acc2, acc3, a_offset) \
        b = Bcol[(a_offset)]; \
        acc0 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4*0+(a_offset)]), b, acc0); \
        acc1 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4*1+(a_offset)]), b, acc1); \
        acc2 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4*2+(a_offset)]), b, acc2); \
        acc3 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4*3+(a_offset)]), b, acc3);


      MADD_STEP4x1(acc00, acc01, acc02, acc03, 2)
      MADD_STEP4x1(acc10, acc11, acc12, acc13, 3)
      Bcol += 4;
      ARow += 4 * a_words_per_iter;

      scalar_t* CPrefetch = Crow;
      _mm_prefetch((char*)(CPrefetch), _MM_HINT_T0); CPrefetch += ldc;
      _mm_prefetch((char*)(CPrefetch), _MM_HINT_T0); CPrefetch += ldc;
      _mm_prefetch((char*)(CPrefetch), _MM_HINT_T0); CPrefetch += ldc;
      _mm_prefetch((char*)(CPrefetch), _MM_HINT_T0); CPrefetch += ldc;

      int k = kSteps - 1;
      do {
        MADD_STEP4x1(acc00, acc01, acc02, acc03, 0)
        MADD_STEP4x1(acc10, acc11, acc12, acc13, 1)
        MADD_STEP4x1(acc00, acc01, acc02, acc03, 2)
        MADD_STEP4x1(acc10, acc11, acc12, acc13, 3)
        Bcol += 4;
        ARow += 4 * a_words_per_iter;
      } while (--k);
      #undef MADD_STEP4x1

      acc00 = MM_ADD_Px(acc00, acc10);
      acc01 = MM_ADD_Px(acc01, acc11);
      acc02 = MM_ADD_Px(acc02, acc12);
      acc03 = MM_ADD_Px(acc03, acc13);

      fp_vector_t  alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
      int_vector_t mask = pPrm->mask_n;
      scalar_t* CCol = Crow;

      if (pPrm->c_option == C_OPTION_UPDATE) {
        MM_MASKSTOREU_Px(CCol, mask, MM_FMADD(acc00, alpha_ps, MM_MASKLOADU_Px(CCol, mask))); CCol += ldc;
        MM_MASKSTOREU_Px(CCol, mask, MM_FMADD(acc01, alpha_ps, MM_MASKLOADU_Px(CCol, mask))); CCol += ldc;
        MM_MASKSTOREU_Px(CCol, mask, MM_FMADD(acc02, alpha_ps, MM_MASKLOADU_Px(CCol, mask))); CCol += ldc;
        MM_MASKSTOREU_Px(CCol, mask, MM_FMADD(acc03, alpha_ps, MM_MASKLOADU_Px(CCol, mask))); CCol += ldc;
      } else if (pPrm->c_option == C_OPTION_REPLACE) {
        MM_MASKSTOREU_Px(CCol, mask, MM_MUL_Px(acc00, alpha_ps)); CCol += ldc;
        MM_MASKSTOREU_Px(CCol, mask, MM_MUL_Px(acc01, alpha_ps)); CCol += ldc;
        MM_MASKSTOREU_Px(CCol, mask, MM_MUL_Px(acc02, alpha_ps)); CCol += ldc;
        MM_MASKSTOREU_Px(CCol, mask, MM_MUL_Px(acc03, alpha_ps)); CCol += ldc;
      } else { // C_OPTION_MULTIPLY
        fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
        MM_MASKSTOREU_Px(CCol, mask, MM_FMADD(acc00, alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(CCol, mask)))); CCol += ldc;
        MM_MASKSTOREU_Px(CCol, mask, MM_FMADD(acc01, alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(CCol, mask)))); CCol += ldc;
        MM_MASKSTOREU_Px(CCol, mask, MM_FMADD(acc02, alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(CCol, mask)))); CCol += ldc;
        MM_MASKSTOREU_Px(CCol, mask, MM_FMADD(acc03, alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(CCol, mask)))); CCol += ldc;
      }
    }

    } break;

    case 3:
    {
    // process 3 rows of A
    const int a_words_per_iter = 3;
    scalar_t* Crow = C;
    const fp_vector_t* Bcol = pPrm->bb;
    for (int b_it = 0; b_it < n_bIters; Crow += B_WORDS_PER_ITER*SIMD_FACTOR, ++b_it) {
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

      b0 = Bcol[0];
      b1 = Bcol[1];
      Bcol += B_WORDS_PER_ITER;
      a = MM_BROADCAST_Sx(&ARow[4*0+1]);
      fp_vector_t acc20 = MM_MUL_Px(a, b0);
      fp_vector_t acc30 = MM_MUL_Px(a, b1);

      a = MM_BROADCAST_Sx(&ARow[4*1+1]);
      fp_vector_t acc21 = MM_MUL_Px(a, b0);
      fp_vector_t acc31 = MM_MUL_Px(a, b1);

      a = MM_BROADCAST_Sx(&ARow[4*2+1]);
      fp_vector_t acc22 = MM_MUL_Px(a, b0);
      fp_vector_t acc32 = MM_MUL_Px(a, b1);

      #define MADD_STEP3x2(acc00, acc10, acc01, acc11, acc02, acc12, a_offset) \
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
      acc12 = MM_FMADD(a, b1, acc12);

      MADD_STEP3x2(acc00, acc10, acc01, acc11, acc02, acc12, 2)
      MADD_STEP3x2(acc20, acc30, acc21, acc31, acc22, acc32, 3)
      ARow += a_words_per_iter*4;

      int k = kSteps-1;
      do {
        MADD_STEP3x2(acc00, acc10, acc01, acc11, acc02, acc12, 0)
        MADD_STEP3x2(acc20, acc30, acc21, acc31, acc22, acc32, 1)
        MADD_STEP3x2(acc00, acc10, acc01, acc11, acc02, acc12, 2)
        MADD_STEP3x2(acc20, acc30, acc21, acc31, acc22, acc32, 3)
        ARow += a_words_per_iter*4;
      } while (--k);
      #undef MADD_STEP3x2

      acc00 = MM_ADD_Px(acc00, acc20);
      acc01 = MM_ADD_Px(acc01, acc21);
      acc02 = MM_ADD_Px(acc02, acc22);
      acc10 = MM_ADD_Px(acc10, acc30);
      acc11 = MM_ADD_Px(acc11, acc31);
      acc12 = MM_ADD_Px(acc12, acc32);

      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
      scalar_t* CCol = Crow;
      if (b_it != pPrm->masked_b_it) {
        if (pPrm->c_option == C_OPTION_UPDATE) {
          #define UPDATE_CCOL(ccol, acc0, acc1) \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*0]), MM_FMADD((acc0), alpha_ps, MM_LOADU_Px(&((ccol)[SIMD_FACTOR*0])))); \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*1]), MM_FMADD((acc1), alpha_ps, MM_LOADU_Px(&((ccol)[SIMD_FACTOR*1]))));

          UPDATE_CCOL(CCol, acc00, acc10); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11); CCol += ldc;
          UPDATE_CCOL(CCol, acc02, acc12);

          #undef UPDATE_CCOL
        } else if (pPrm->c_option == C_OPTION_REPLACE) {
          #define UPDATE_CCOL(ccol, acc0, acc1) \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*0]), MM_MUL_Px((acc0), alpha_ps)); \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*1]), MM_MUL_Px((acc1), alpha_ps));

          UPDATE_CCOL(CCol, acc00, acc10); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11); CCol += ldc;
          UPDATE_CCOL(CCol, acc02, acc12);

          #undef UPDATE_CCOL
        } else { // C_OPTION_MULTIPLY
          fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
          #define UPDATE_CCOL(ccol, acc0, acc1) \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*0]), MM_FMADD((acc0), alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px(&((ccol)[SIMD_FACTOR*0]))))); \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*1]), MM_FMADD((acc1), alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px(&((ccol)[SIMD_FACTOR*1])))));

          UPDATE_CCOL(CCol, acc00, acc10); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11); CCol += ldc;
          UPDATE_CCOL(CCol, acc02, acc12);

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
          UPDATE_CCOL(CCol, acc02, acc12);


          #undef UPDATE_CCOL
        } else if (pPrm->c_option == C_OPTION_REPLACE) {
          #define UPDATE_CCOL(ccol, acc0, acc1) \
          MM_STOREU_Px    (&((ccol)[SIMD_FACTOR*0]),       MM_MUL_Px((acc0), alpha_ps)); \
          MM_MASKSTOREU_Px(&((ccol)[SIMD_FACTOR*1]), mask, MM_MUL_Px((acc1), alpha_ps));

          UPDATE_CCOL(CCol, acc00, acc10); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11); CCol += ldc;
          UPDATE_CCOL(CCol, acc02, acc12);

          #undef UPDATE_CCOL
        } else { // C_OPTION_MULTIPLY
          fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
          #define UPDATE_CCOL(ccol, acc0, acc1) \
          MM_STOREU_Px    (&((ccol)[SIMD_FACTOR*0]),       MM_FMADD((acc0), alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px    (&((ccol)[SIMD_FACTOR*0]))))); \
          MM_MASKSTOREU_Px(&((ccol)[SIMD_FACTOR*1]), mask, MM_FMADD((acc1), alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(&((ccol)[SIMD_FACTOR*1]), mask))));

          UPDATE_CCOL(CCol, acc00, acc10); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11); CCol += ldc;
          UPDATE_CCOL(CCol, acc02, acc12);

          #undef UPDATE_CCOL
        }
      }
    }

    if (pPrm->hasMinor) {
      // multiply of 3 rows of A by one SIMD column of B
      const scalar_t* ARow = A;
      fp_vector_t b;

      b = Bcol[0];
      fp_vector_t acc00 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[4*0+0]), b);
      fp_vector_t acc01 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[4*1+0]), b);
      fp_vector_t acc02 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[4*2+0]), b);

      b = Bcol[1];
      fp_vector_t acc10 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[4*0+1]), b);
      fp_vector_t acc11 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[4*1+1]), b);
      fp_vector_t acc12 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[4*2+1]), b);

      #define MADD_STEP3x1(acc0, acc1, acc2, a_offset) \
        b = Bcol[(a_offset)]; \
        acc0 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4*0+(a_offset)]), b, acc0); \
        acc1 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4*1+(a_offset)]), b, acc1); \
        acc2 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4*2+(a_offset)]), b, acc2);


      MADD_STEP3x1(acc00, acc01, acc02, 2)
      MADD_STEP3x1(acc10, acc11, acc12, 3)
      Bcol += 4;
      ARow += 4 * a_words_per_iter;

      scalar_t* CPrefetch = Crow;
      _mm_prefetch((char*)(CPrefetch), _MM_HINT_T0); CPrefetch += ldc;
      _mm_prefetch((char*)(CPrefetch), _MM_HINT_T0); CPrefetch += ldc;
      _mm_prefetch((char*)(CPrefetch), _MM_HINT_T0); CPrefetch += ldc;

      int k = kSteps - 1;
      do {
        MADD_STEP3x1(acc00, acc01, acc02, 0)
        MADD_STEP3x1(acc10, acc11, acc12, 1)
        MADD_STEP3x1(acc00, acc01, acc02, 2)
        MADD_STEP3x1(acc10, acc11, acc12, 3)
        Bcol += 4;
        ARow += 4 * a_words_per_iter;
      } while (--k);
      #undef MADD_STEP3x1

      acc00 = MM_ADD_Px(acc00, acc10);
      acc01 = MM_ADD_Px(acc01, acc11);
      acc02 = MM_ADD_Px(acc02, acc12);

      fp_vector_t  alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
      int_vector_t mask = pPrm->mask_n;
      scalar_t* CCol = Crow;

      if (pPrm->c_option == C_OPTION_UPDATE) {
        MM_MASKSTOREU_Px(CCol, mask, MM_FMADD(acc00, alpha_ps, MM_MASKLOADU_Px(CCol, mask))); CCol += ldc;
        MM_MASKSTOREU_Px(CCol, mask, MM_FMADD(acc01, alpha_ps, MM_MASKLOADU_Px(CCol, mask))); CCol += ldc;
        MM_MASKSTOREU_Px(CCol, mask, MM_FMADD(acc02, alpha_ps, MM_MASKLOADU_Px(CCol, mask))); CCol += ldc;
      } else if (pPrm->c_option == C_OPTION_REPLACE) {
        MM_MASKSTOREU_Px(CCol, mask, MM_MUL_Px(acc00, alpha_ps)); CCol += ldc;
        MM_MASKSTOREU_Px(CCol, mask, MM_MUL_Px(acc01, alpha_ps)); CCol += ldc;
        MM_MASKSTOREU_Px(CCol, mask, MM_MUL_Px(acc02, alpha_ps)); CCol += ldc;
      } else { // C_OPTION_MULTIPLY
        fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
        MM_MASKSTOREU_Px(CCol, mask, MM_FMADD(acc00, alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(CCol, mask)))); CCol += ldc;
        MM_MASKSTOREU_Px(CCol, mask, MM_FMADD(acc01, alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(CCol, mask)))); CCol += ldc;
        MM_MASKSTOREU_Px(CCol, mask, MM_FMADD(acc02, alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(CCol, mask)))); CCol += ldc;
      }
    }

    } break;

    case 2:
    {
    // process 2 rows of A
    const int a_words_per_iter = 2;
    scalar_t* Crow = C;
    const fp_vector_t* Bcol = pPrm->bb;
    for (int b_it = 0; b_it < n_bIters; Crow += B_WORDS_PER_ITER*SIMD_FACTOR, ++b_it) {
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

      b0 = Bcol[0];
      b1 = Bcol[1];
      Bcol += B_WORDS_PER_ITER;
      a = MM_BROADCAST_Sx(&ARow[4*0+1]);
      fp_vector_t acc20 = MM_MUL_Px(a, b0);
      fp_vector_t acc30 = MM_MUL_Px(a, b1);

      a = MM_BROADCAST_Sx(&ARow[4*1+1]);
      fp_vector_t acc21 = MM_MUL_Px(a, b0);
      fp_vector_t acc31 = MM_MUL_Px(a, b1);

      #define MADD_STEP2x2(acc00, acc10, acc01, acc11, a_offset) \
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
      acc11 = MM_FMADD(a, b1, acc11);

      MADD_STEP2x2(acc00, acc10, acc01, acc11, 2)
      MADD_STEP2x2(acc20, acc30, acc21, acc31, 3)
      ARow += a_words_per_iter*4;

      int k = kSteps-1;
      do {
        MADD_STEP2x2(acc00, acc10, acc01, acc11, 0)
        MADD_STEP2x2(acc20, acc30, acc21, acc31, 1)
        MADD_STEP2x2(acc00, acc10, acc01, acc11, 2)
        MADD_STEP2x2(acc20, acc30, acc21, acc31, 3)
        ARow += a_words_per_iter*4;
      } while (--k);
      #undef MADD_STEP2x2

      acc00 = MM_ADD_Px(acc00, acc20);
      acc01 = MM_ADD_Px(acc01, acc21);
      acc10 = MM_ADD_Px(acc10, acc30);
      acc11 = MM_ADD_Px(acc11, acc31);

      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
      scalar_t* CCol = Crow;
      if (b_it != pPrm->masked_b_it) {
        if (pPrm->c_option == C_OPTION_UPDATE) {
          #define UPDATE_CCOL(ccol, acc0, acc1) \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*0]), MM_FMADD((acc0), alpha_ps, MM_LOADU_Px(&((ccol)[SIMD_FACTOR*0])))); \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*1]), MM_FMADD((acc1), alpha_ps, MM_LOADU_Px(&((ccol)[SIMD_FACTOR*1]))));

          UPDATE_CCOL(CCol, acc00, acc10); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11);

          #undef UPDATE_CCOL
        } else if (pPrm->c_option == C_OPTION_REPLACE) {
          #define UPDATE_CCOL(ccol, acc0, acc1) \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*0]), MM_MUL_Px((acc0), alpha_ps)); \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*1]), MM_MUL_Px((acc1), alpha_ps));

          UPDATE_CCOL(CCol, acc00, acc10); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11);

          #undef UPDATE_CCOL
        } else { // C_OPTION_MULTIPLY
          fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
          #define UPDATE_CCOL(ccol, acc0, acc1) \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*0]), MM_FMADD((acc0), alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px(&((ccol)[SIMD_FACTOR*0]))))); \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*1]), MM_FMADD((acc1), alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px(&((ccol)[SIMD_FACTOR*1])))));

          UPDATE_CCOL(CCol, acc00, acc10); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11);

          #undef UPDATE_CCOL
        }
      } else {
        int_vector_t mask = pPrm->mask_n;
        if (pPrm->c_option == C_OPTION_UPDATE) {
          #define UPDATE_CCOL(ccol, acc0, acc1) \
          MM_STOREU_Px    (&((ccol)[SIMD_FACTOR*0]),       MM_FMADD((acc0), alpha_ps, MM_LOADU_Px    (&((ccol)[SIMD_FACTOR*0])))); \
          MM_MASKSTOREU_Px(&((ccol)[SIMD_FACTOR*1]), mask, MM_FMADD((acc1), alpha_ps, MM_MASKLOADU_Px(&((ccol)[SIMD_FACTOR*1]), mask)));

          UPDATE_CCOL(CCol, acc00, acc10); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11);


          #undef UPDATE_CCOL
        } else if (pPrm->c_option == C_OPTION_REPLACE) {
          #define UPDATE_CCOL(ccol, acc0, acc1) \
          MM_STOREU_Px    (&((ccol)[SIMD_FACTOR*0]),       MM_MUL_Px((acc0), alpha_ps)); \
          MM_MASKSTOREU_Px(&((ccol)[SIMD_FACTOR*1]), mask, MM_MUL_Px((acc1), alpha_ps));

          UPDATE_CCOL(CCol, acc00, acc10); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11);

          #undef UPDATE_CCOL
        } else { // C_OPTION_MULTIPLY
          fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
          #define UPDATE_CCOL(ccol, acc0, acc1) \
          MM_STOREU_Px    (&((ccol)[SIMD_FACTOR*0]),       MM_FMADD((acc0), alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px    (&((ccol)[SIMD_FACTOR*0]))))); \
          MM_MASKSTOREU_Px(&((ccol)[SIMD_FACTOR*1]), mask, MM_FMADD((acc1), alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(&((ccol)[SIMD_FACTOR*1]), mask))));

          UPDATE_CCOL(CCol, acc00, acc10); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11);

          #undef UPDATE_CCOL
        }
      }
    }

    if (pPrm->hasMinor) {
       // multiply of 2 rows of A by one SIMD column of B
      const scalar_t* ARow = A;
      fp_vector_t b;

      b = Bcol[0];
      fp_vector_t acc00 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[4*0+0]), b);
      fp_vector_t acc01 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[4*1+0]), b);

      b = Bcol[1];
      fp_vector_t acc10 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[4*0+1]), b);
      fp_vector_t acc11 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[4*1+1]), b);

      #define MADD_STEP2x1(acc0, acc1, a_offset) \
        b = Bcol[(a_offset)]; \
        acc0 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4*0+(a_offset)]), b, acc0); \
        acc1 = MM_FMADD(MM_BROADCAST_Sx(&ARow[4*1+(a_offset)]), b, acc1);


      MADD_STEP2x1(acc00, acc01, 2)
      MADD_STEP2x1(acc10, acc11, 3)
      Bcol += 4;
      ARow += 4 * a_words_per_iter;

      scalar_t* CPrefetch = Crow;
      _mm_prefetch((char*)(CPrefetch), _MM_HINT_T0); CPrefetch += ldc;
      _mm_prefetch((char*)(CPrefetch), _MM_HINT_T0); CPrefetch += ldc;

      int k = kSteps - 1;
      do {
        MADD_STEP2x1(acc00, acc01, 0)
        MADD_STEP2x1(acc10, acc11, 1)
        MADD_STEP2x1(acc00, acc01, 2)
        MADD_STEP2x1(acc10, acc11, 3)
        Bcol += 4;
        ARow += 4 * a_words_per_iter;
      } while (--k);
      #undef MADD_STEP2x1

      acc00 = MM_ADD_Px(acc00, acc10);
      acc01 = MM_ADD_Px(acc01, acc11);

      fp_vector_t  alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
      int_vector_t mask = pPrm->mask_n;
      scalar_t* CCol = Crow;
      if (pPrm->c_option == C_OPTION_UPDATE) {
        MM_MASKSTOREU_Px(CCol, mask, MM_FMADD(acc00, alpha_ps, MM_MASKLOADU_Px(CCol, mask))); CCol += ldc;
        MM_MASKSTOREU_Px(CCol, mask, MM_FMADD(acc01, alpha_ps, MM_MASKLOADU_Px(CCol, mask))); CCol += ldc;
      } else if (pPrm->c_option == C_OPTION_REPLACE) {
        MM_MASKSTOREU_Px(CCol, mask, MM_MUL_Px(acc00, alpha_ps)); CCol += ldc;
        MM_MASKSTOREU_Px(CCol, mask, MM_MUL_Px(acc01, alpha_ps)); CCol += ldc;
      } else { // C_OPTION_MULTIPLY
        fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
        MM_MASKSTOREU_Px(CCol, mask, MM_FMADD(acc00, alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(CCol, mask)))); CCol += ldc;
        MM_MASKSTOREU_Px(CCol, mask, MM_FMADD(acc01, alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(CCol, mask)))); CCol += ldc;
      }
    }

    } break;

    default:
    {
      // process 1 row of A
      scalar_t* Crow = C;
      const fp_vector_t* Bcol = pPrm->bb;
      for (int b_it = 0; b_it < n_bIters; Crow += B_WORDS_PER_ITER*SIMD_FACTOR, ++b_it) {
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

      if (pPrm->hasMinor) {
        // multiply of one row of A by one SIMD column of B
        const scalar_t* ARow = A;

        fp_vector_t acc00 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[0]), Bcol[0]);
        fp_vector_t acc01 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[1]), Bcol[1]);
        fp_vector_t acc02 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[2]), Bcol[2]);
        fp_vector_t acc03 = MM_MUL_Px(MM_BROADCAST_Sx(&ARow[3]), Bcol[3]);
        Bcol += 4;
        ARow += 4;

        _mm_prefetch((char*)(Crow), _MM_HINT_T0);

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
          MM_MASKSTOREU_Px(Crow, mask, MM_FMADD(acc00, alpha_ps, MM_MASKLOADU_Px(Crow, mask)));
        } else if (pPrm->c_option == C_OPTION_REPLACE) {
          MM_MASKSTOREU_Px(Crow, mask, MM_MUL_Px(acc00, alpha_ps));
        } else { // C_OPTION_MULTIPLY
          fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
          MM_MASKSTOREU_Px(Crow, mask, MM_FMADD(acc00, alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(Crow, mask))));
        }
      }

    } break;
  }

  #undef PrefetchLines
}

static void CopyAndTransposeBMjx2(
  noncblas_sgemm_prm_t* pPrm,
  const scalar_t*       B,
  int                   nRows)
{
  fp_vector_t* dstCol = pPrm->bb;
  int ldbb = ((unsigned)(nRows+3)/4)*4*B_WORDS_PER_ITER;
  int ldb = pPrm->ldb;
  for (int r = 0; r < nRows; ++r) {
    fp_vector_t w00 = MM_LOADU_Px(&B[0*SIMD_FACTOR]);
    fp_vector_t w01 = MM_LOADU_Px(&B[1*SIMD_FACTOR]);
    fp_vector_t w10 = MM_LOADU_Px(&B[2*SIMD_FACTOR]);
    fp_vector_t w11 = MM_LOADU_Px(&B[3*SIMD_FACTOR]);
    fp_vector_t w20 = MM_LOADU_Px(&B[4*SIMD_FACTOR]);
    fp_vector_t w21 = MM_LOADU_Px(&B[5*SIMD_FACTOR]);
    fp_vector_t w30 = MM_LOADU_Px(&B[6*SIMD_FACTOR]);
    fp_vector_t w31 = MM_LOADU_Px(&B[7*SIMD_FACTOR]);
    B += ldb;
    dstCol[0]        = w00;
    dstCol[1]        = w01;
    dstCol[ldbb*1+0] = w10;
    dstCol[ldbb*1+1] = w11;
    dstCol[ldbb*2+0] = w20;
    dstCol[ldbb*2+1] = w21;
    dstCol[ldbb*3+0] = w30;
    dstCol[ldbb*3+1] = w31;
    dstCol += B_WORDS_PER_ITER;
  }
}

static void CopyAndTransposeBRem(
  noncblas_sgemm_prm_t* pPrm,
  const scalar_t*       B,
  int                   nRows,
  int                   nw)
{
  fp_vector_t* dstCol = pPrm->bb;
  int_vector_t mask = pPrm->mask_n;
  int ldbb = ((unsigned)(nRows+3)/4)*4*B_WORDS_PER_ITER;
  int ldb = pPrm->ldb;
  switch (nw) {
    case 1:
    {
      for (int r = 0; r < nRows; B += ldb, ++r)
        dstCol[r] = MM_MASKLOADU_Px(B, mask);
    } break;

    case 2:
    {
      for (int r = 0; r < nRows; ++r) {
        fp_vector_t w00 = MM_LOADU_Px(&B[0*SIMD_FACTOR]);
        fp_vector_t w01 = MM_MASKLOADU_Px(&B[1*SIMD_FACTOR], mask);
        B += ldb;
        dstCol[0]        = w00;
        dstCol[1]        = w01;
        dstCol += B_WORDS_PER_ITER;
      }
    } break;

    case 3:
    {
      fp_vector_t* dstLastCol = &dstCol[ldbb*1+0];
      for (int r = 0; r < nRows; ++r) {
        fp_vector_t w00 = MM_LOADU_Px(&B[0*SIMD_FACTOR]);
        fp_vector_t w01 = MM_LOADU_Px(&B[1*SIMD_FACTOR]);
        fp_vector_t w10 = MM_MASKLOADU_Px(&B[2*SIMD_FACTOR], mask);
        B += ldb;
        dstCol[0]        = w00;
        dstCol[1]        = w01;
        dstCol += B_WORDS_PER_ITER;
        dstLastCol[0]    = w10;
        dstLastCol += 1;
      }
    } break;

    case 4:
    {
      for (int r = 0; r < nRows; ++r) {
        fp_vector_t w00 = MM_LOADU_Px(&B[0*SIMD_FACTOR]);
        fp_vector_t w01 = MM_LOADU_Px(&B[1*SIMD_FACTOR]);
        fp_vector_t w10 = MM_LOADU_Px(&B[2*SIMD_FACTOR]);
        fp_vector_t w11 = MM_MASKLOADU_Px(&B[3*SIMD_FACTOR], mask);
        B += ldb;
        dstCol[0]        = w00;
        dstCol[1]        = w01;
        dstCol[ldbb*1+0] = w10;
        dstCol[ldbb*1+1] = w11;
        dstCol += B_WORDS_PER_ITER;
      }
    } break;

    case 5:
    {
      fp_vector_t* dstLastCol = &dstCol[ldbb*2+0];
      for (int r = 0; r < nRows; ++r) {
        fp_vector_t w00 = MM_LOADU_Px(&B[0*SIMD_FACTOR]);
        fp_vector_t w01 = MM_LOADU_Px(&B[1*SIMD_FACTOR]);
        fp_vector_t w10 = MM_LOADU_Px(&B[2*SIMD_FACTOR]);
        fp_vector_t w11 = MM_LOADU_Px(&B[3*SIMD_FACTOR]);
        fp_vector_t w20 = MM_MASKLOADU_Px(&B[4*SIMD_FACTOR], mask);
        B += ldb;
        dstCol[0]        = w00;
        dstCol[1]        = w01;
        dstCol[ldbb*1+0] = w10;
        dstCol[ldbb*1+1] = w11;
        dstCol += B_WORDS_PER_ITER;
        dstLastCol[0]    = w20;
        dstLastCol += 1;
      }
    } break;

    case 6:
    {
      for (int r = 0; r < nRows; ++r) {
        fp_vector_t w00 = MM_LOADU_Px(&B[0*SIMD_FACTOR]);
        fp_vector_t w01 = MM_LOADU_Px(&B[1*SIMD_FACTOR]);
        fp_vector_t w10 = MM_LOADU_Px(&B[2*SIMD_FACTOR]);
        fp_vector_t w11 = MM_LOADU_Px(&B[3*SIMD_FACTOR]);
        fp_vector_t w20 = MM_LOADU_Px(&B[4*SIMD_FACTOR]);
        fp_vector_t w21 = MM_MASKLOADU_Px(&B[5*SIMD_FACTOR], mask);
        B += ldb;
        dstCol[0]        = w00;
        dstCol[1]        = w01;
        dstCol[ldbb*1+0] = w10;
        dstCol[ldbb*1+1] = w11;
        dstCol[ldbb*2+0] = w20;
        dstCol[ldbb*2+1] = w21;
        dstCol += B_WORDS_PER_ITER;
      }
    } break;

    case 7:
    {
      fp_vector_t* dstLastCol = &dstCol[ldbb*3+0];
      for (int r = 0; r < nRows; ++r) {
        fp_vector_t w00 = MM_LOADU_Px(&B[0*SIMD_FACTOR]);
        fp_vector_t w01 = MM_LOADU_Px(&B[1*SIMD_FACTOR]);
        fp_vector_t w10 = MM_LOADU_Px(&B[2*SIMD_FACTOR]);
        fp_vector_t w11 = MM_LOADU_Px(&B[3*SIMD_FACTOR]);
        fp_vector_t w20 = MM_LOADU_Px(&B[4*SIMD_FACTOR]);
        fp_vector_t w21 = MM_LOADU_Px(&B[5*SIMD_FACTOR]);
        fp_vector_t w30 = MM_MASKLOADU_Px(&B[6*SIMD_FACTOR], mask);
        B += ldb;
        dstCol[0]        = w00;
        dstCol[1]        = w01;
        dstCol[ldbb*1+0] = w10;
        dstCol[ldbb*1+1] = w11;
        dstCol[ldbb*2+0] = w20;
        dstCol[ldbb*2+1] = w21;
        dstCol += B_WORDS_PER_ITER;
        dstLastCol[0]    = w30;
        dstLastCol += 1;
      }
    } break;

    case 8:
    {
      for (int r = 0; r < nRows; ++r) {
        fp_vector_t w00 = MM_LOADU_Px(&B[0*SIMD_FACTOR]);
        fp_vector_t w01 = MM_LOADU_Px(&B[1*SIMD_FACTOR]);
        fp_vector_t w10 = MM_LOADU_Px(&B[2*SIMD_FACTOR]);
        fp_vector_t w11 = MM_LOADU_Px(&B[3*SIMD_FACTOR]);
        fp_vector_t w20 = MM_LOADU_Px(&B[4*SIMD_FACTOR]);
        fp_vector_t w21 = MM_LOADU_Px(&B[5*SIMD_FACTOR]);
        fp_vector_t w30 = MM_LOADU_Px(&B[6*SIMD_FACTOR]);
        fp_vector_t w31 = MM_MASKLOADU_Px(&B[7*SIMD_FACTOR], mask);
        B += ldb;
        dstCol[0]        = w00;
        dstCol[1]        = w01;
        dstCol[ldbb*1+0] = w10;
        dstCol[ldbb*1+1] = w11;
        dstCol[ldbb*2+0] = w20;
        dstCol[ldbb*2+1] = w21;
        dstCol[ldbb*3+0] = w30;
        dstCol[ldbb*3+1] = w31;
        dstCol += B_WORDS_PER_ITER;
      }
    } break;
  }
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

  if (pPrm->mRem == 0)
    return;

  const scalar_t *src = A;
  int k = (unsigned)n_cols / 4;
  switch (pPrm->mRem) {
    case 4:
    {
      // interleave 4 rows
      const int a_words_per_iter = 4;
      do {
        fp_vector4_t a0 = MM_LOADU4_Px(&src[0]);
        fp_vector4_t a1 = MM_LOADU4_Px(&src[lda1]);
        fp_vector4_t a2 = MM_LOADU4_Px(&src[lda2]);
        fp_vector4_t a3 = MM_LOADU4_Px(&src[lda3]);
        src += 4;
        dst[0] = a0;
        dst[1] = a1;
        dst[2] = a2;
        dst[3] = a3;
        dst += a_words_per_iter;
      } while (--k);
      if ((unsigned)n_cols % 4 != 0) {
        int_vector4_t mask = pPrm->mask_k;
        fp_vector4_t a0 = MM_MASKLOADU4_Px(&src[0],    mask);
        fp_vector4_t a1 = MM_MASKLOADU4_Px(&src[lda1], mask);
        fp_vector4_t a2 = MM_MASKLOADU4_Px(&src[lda2], mask);
        fp_vector4_t a3 = MM_MASKLOADU4_Px(&src[lda3], mask);
        dst[0] = a0;
        dst[1] = a1;
        dst[2] = a2;
        dst[3] = a3;
      }
    } break;

    case 3:
    {
      // interleave 3 rows
      const int a_words_per_iter = 3;
      do {
        fp_vector4_t a0 = MM_LOADU4_Px(&src[0]);
        fp_vector4_t a1 = MM_LOADU4_Px(&src[lda1]);
        fp_vector4_t a2 = MM_LOADU4_Px(&src[lda2]);
        src += 4;
        dst[0] = a0;
        dst[1] = a1;
        dst[2] = a2;
        dst += a_words_per_iter;
      } while (--k);
      if ((unsigned)n_cols % 4 != 0) {
        int_vector4_t mask = pPrm->mask_k;
        fp_vector4_t a0 = MM_MASKLOADU4_Px(&src[0],    mask);
        fp_vector4_t a1 = MM_MASKLOADU4_Px(&src[lda1], mask);
        fp_vector4_t a2 = MM_MASKLOADU4_Px(&src[lda2], mask);
        dst[0] = a0;
        dst[1] = a1;
        dst[2] = a2;
      }
    } break;

    case 2:
    {
      // interleave 2 rows
      const int a_words_per_iter = 2;
      do {
        fp_vector4_t a0 = MM_LOADU4_Px(&src[0]);
        fp_vector4_t a1 = MM_LOADU4_Px(&src[lda1]);
        src += 4;
        dst[0] = a0;
        dst[1] = a1;
        dst += a_words_per_iter;
      } while (--k);
      if ((unsigned)n_cols % 4 != 0) {
        int_vector4_t mask = pPrm->mask_k;
        fp_vector4_t a0 = MM_MASKLOADU4_Px(&src[0],    mask);
        fp_vector4_t a1 = MM_MASKLOADU4_Px(&src[lda1], mask);
        dst[0] = a0;
        dst[1] = a1;
      }
    } break;

    default:
    {
      // single row not interleaved
      do {
        dst[0] = MM_LOADU4_Px(&src[0]);
        src += 4;
        dst += 1;
      } while (--k);
      if ((unsigned)n_cols % 4 != 0) {
        dst[0] = MM_MASKLOADU4_Px(&src[0], pPrm->mask_k);
      }
    } break;
  }
}

extern uint64_t dbg_tt;
#include <stdio.h>
#include <x86intrin.h>

#ifdef  NONCBLAS_SGEMM_TUNE
//int st_m_step = 0;
int st_k_step = 0;
#endif

// N>SIMD_FACTOR
static void noncblas_sgemm_wide_n(noncblas_sgemm_prm_t* pPrm)
{
  int nMj       = pPrm->N / n_step;
  unsigned nRem = pPrm->N % n_step;
  int nwRem     = (nRem+SIMD_FACTOR-1) / SIMD_FACTOR;
  int nwRemMj   = (unsigned)nwRem / B_WORDS_PER_ITER;
  int nwRemMn   = (unsigned)nwRem % B_WORDS_PER_ITER;

  int m_step_nom = pPrm->M;
  if (m_step_nom > (M_STEP/2)*3) {
    int m_Nsteps = (m_step_nom-1)/M_STEP + 1;
    m_step_nom = ((m_step_nom-1)/(m_Nsteps*A_WORDS_PER_ITER) + 1) * A_WORDS_PER_ITER;
  }

  // calculate k_step
  int k_step;
  #ifdef  NONCBLAS_SGEMM_TUNE
  if (st_k_step > 0) {
    k_step = K;
    if (st_k_step+3 < K) {
      k_step = ((unsigned)(st_k_step+3)/4)*4;
    }
  } else {
  #endif
  k_step = pPrm->K;
  if (k_step > K_STEP_MAX) {
    int k_Nsteps = (k_step-1)/K_STEP_NOM + 1;
    k_step = ((k_step-1)/(k_Nsteps*4) + 1) * 4;
  }
  #ifdef  NONCBLAS_SGEMM_TUNE
  }
  #endif

  const int nMj_h = (unsigned)nMj/2;
  const int nMj_r = (unsigned)nMj%2;
  const int nwRem_ex = SIMD_ELEM_PEC_COL_MJ*nMj_r+nwRem;

  const int CACHE_LINE_SZ = 64;
  const int k_step_ex = ((unsigned)(k_step-1)/4 + 1)*4;
  const int bb_nw = nMj_h == 0 ? nwRem_ex : SIMD_ELEM_PEC_COL_MJ*2;
  const int bb_sz = (unsigned)(bb_nw*k_step_ex*sizeof(fp_vector_t)-1)/CACHE_LINE_SZ+1;
  const int aa_sz = (unsigned)(m_step_nom*k_step_ex*sizeof(scalar_t) - 1)/CACHE_LINE_SZ+1;
  const int workBufSz = aa_sz + bb_sz;
  // I didn't find a standard portable way to allocate 32-byte aligned buffer
  // So I am doing it in hackish, but reliable way
  char* workBufAlloc = malloc((workBufSz+1)*CACHE_LINE_SZ);
  uintptr_t workBufAdj = (0-(uintptr_t)(workBufAlloc)) % CACHE_LINE_SZ;
  char* workBuf = workBufAlloc+workBufAdj;

  pPrm->aa = (fp_vector4_t*)(workBuf+0);
  fp_vector_t* bb = (fp_vector_t*) (workBuf+aa_sz*CACHE_LINE_SZ);

  int nwRemMj_masked_b_it = -1;
  nRem %= SIMD_FACTOR;
  if (nRem > 0 && nwRemMn == 0)
    nwRemMj_masked_b_it =  nwRemMj - 1;


  memset(&pPrm->mask_k, -1, sizeof(pPrm->mask_k));
  unsigned kRem = pPrm->K % 4;
  if (kRem > 0) { // mask off elements of rightmost SIMD word in A
    memset(&pPrm->mask_k, 0, sizeof(pPrm->mask_k));
    memset((char*)&pPrm->mask_k, -1, sizeof(scalar_t)*kRem);
  }

  // printf("nMj=%d, nwRemMn=%d, nwRemMj=%d nRem=%d nwRemMj_masked_b_it=%d\n", nMj, nwRemMn, nwRemMj, nRem, nwRemMj_masked_b_it);
  uint64_t tt = 0;

  for (int k = 0; k < pPrm->K; k += k_step) {
    pPrm->c_option = C_OPTION_UPDATE;
    if (k==0 && pPrm->beta != 1.0f)
      pPrm->c_option = (pPrm->beta == 0) ? C_OPTION_REPLACE : C_OPTION_MULTIPLY;
    int delta_k = pPrm->K - k;
    if (delta_k > k_step) {
      if ((delta_k-k_step)*2 < k_step)
        k_step = ((unsigned)(delta_k-1)/(4*2) + 1)*4;
      delta_k = k_step;
    }
    const int kSteps = (unsigned)(delta_k-1)/4 + 1;

    int m_step = m_step_nom;
    for (int m = 0; m < pPrm->M; m += m_step) {
      int delta_m = pPrm->M - m;
      if (delta_m > m_step) {
        if ((delta_m - m_step)*2 < m_step)
          m_step = ((unsigned)(delta_m-1)/(A_WORDS_PER_ITER*2) + 1)*A_WORDS_PER_ITER;
        delta_m = m_step;
      }

      pPrm->mDiv = delta_m / A_WORDS_PER_ITER;
      pPrm->mRem = delta_m - pPrm->mDiv*A_WORDS_PER_ITER;

      pPrm->masked_b_it = -1;          // all words in use
      CopyAndInterleaveA(pPrm, &pPrm->A[m*pPrm->lda+k], delta_k);

      scalar_t *Crow = &pPrm->C[m*pPrm->ldc];
      const scalar_t *Brow = &pPrm->B[k*pPrm->ldb];

      int ldbb = kSteps*4*B_WORDS_PER_ITER;
      pPrm->hasMinor = 0;
      for (int ni = 0; ni < nMj_h; ++ni) {
        // process two full-width major rectangles
        int n = ni * n_step * 2;
        uint64_t t0 = __rdtsc();
        pPrm->bb = bb;
        CopyAndTransposeBMjx2(pPrm, &Brow[n], delta_k);
        uint64_t t1 = __rdtsc();
        tt += t1 - t0;

        pPrm->bb = bb;
        fma256_noncblas_sgemm_core(pPrm, &Crow[n], N_STEP_MULTIPLIER, kSteps);
        pPrm->bb += ldbb*N_STEP_MULTIPLIER;
        fma256_noncblas_sgemm_core(pPrm, &Crow[n+n_step], N_STEP_MULTIPLIER, kSteps);
      }

      if (nwRem_ex != 0) {
        int n = nMj_h * n_step * 2;
        uint64_t t0 = __rdtsc();
        pPrm->bb = bb;
        CopyAndTransposeBRem(pPrm, &Brow[n], delta_k, nwRem_ex);
        uint64_t t1 = __rdtsc();
        tt += t1 - t0;
        if (nMj_r != 0) {
          if (nwRemMj == 0)
            pPrm->masked_b_it = nwRemMj_masked_b_it;
          // process last full-width major rectangles
          fma256_noncblas_sgemm_core(pPrm, &Crow[n], N_STEP_MULTIPLIER, kSteps);
          n += n_step;
          pPrm->bb += ldbb*N_STEP_MULTIPLIER;
        }
        if ((nwRemMj | nwRemMn) != 0) {
          pPrm->masked_b_it = nwRemMj_masked_b_it;
          pPrm->hasMinor = nwRemMn;
          fma256_noncblas_sgemm_core(pPrm, &Crow[n], nwRemMj, kSteps);
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

static void noncblas_sgemm_k1(noncblas_sgemm_prm_t* pPrm)
{
  if (pPrm->beta == 0) {
    if (pPrm->alpha == 1.0f) {
      unsigned nDiv = pPrm->N / (SIMD_FACTOR*8);
      while (nDiv != 0) {
        const scalar_t *B = pPrm->B;
        fp_vector_t b0 = MM_LOADU_Px(B); B += SIMD_FACTOR;
        fp_vector_t b1 = MM_LOADU_Px(B); B += SIMD_FACTOR;
        fp_vector_t b2 = MM_LOADU_Px(B); B += SIMD_FACTOR;
        fp_vector_t b3 = MM_LOADU_Px(B); B += SIMD_FACTOR;
        fp_vector_t b4 = MM_LOADU_Px(B); B += SIMD_FACTOR;
        fp_vector_t b5 = MM_LOADU_Px(B); B += SIMD_FACTOR;
        fp_vector_t b6 = MM_LOADU_Px(B); B += SIMD_FACTOR;
        fp_vector_t b7 = MM_LOADU_Px(B); B += SIMD_FACTOR;

        const scalar_t *A = pPrm->A;
        int lda = pPrm->lda;
        scalar_t *C = pPrm->C;
        int ldc = pPrm->ldc;
        unsigned M = pPrm->M;
        do {
          fp_vector_t a = MM_BROADCAST_Sx(A); A += lda;
          MM_STOREU_Px(C+SIMD_FACTOR*0, MM_MUL_Px(b0, a));
          MM_STOREU_Px(C+SIMD_FACTOR*1, MM_MUL_Px(b1, a));
          MM_STOREU_Px(C+SIMD_FACTOR*2, MM_MUL_Px(b2, a));
          MM_STOREU_Px(C+SIMD_FACTOR*3, MM_MUL_Px(b3, a));
          MM_STOREU_Px(C+SIMD_FACTOR*4, MM_MUL_Px(b4, a));
          MM_STOREU_Px(C+SIMD_FACTOR*5, MM_MUL_Px(b5, a));
          MM_STOREU_Px(C+SIMD_FACTOR*6, MM_MUL_Px(b6, a));
          MM_STOREU_Px(C+SIMD_FACTOR*7, MM_MUL_Px(b7, a));
          C += ldc;
        } while (--M);

        pPrm->B += SIMD_FACTOR*8;
        pPrm->C += SIMD_FACTOR*8;
        --nDiv;
      }

      unsigned nRem = pPrm->N % (SIMD_FACTOR*8);
      nDiv = nRem / SIMD_FACTOR;
      const scalar_t *B = pPrm->B;
      while (nDiv != 0) {
        fp_vector_t b = MM_LOADU_Px(B); B += SIMD_FACTOR;

        const scalar_t *A = pPrm->A;
        int lda = pPrm->lda;
        scalar_t *C = pPrm->C;
        int ldc = pPrm->ldc;
        unsigned M = pPrm->M;
        do {
          fp_vector_t a = MM_BROADCAST_Sx(A); A += lda;
          MM_STOREU_Px(C, MM_MUL_Px(b, a));   C += ldc;
        } while (--M);

        pPrm->C += SIMD_FACTOR;
        --nDiv;
      }

      if (pPrm->N % SIMD_FACTOR) {
        int_vector_t mask_n = pPrm->mask_n;
        fp_vector_t b = MM_MASKLOADU_Px(B, mask_n);

        const scalar_t *A = pPrm->A;
        int lda = pPrm->lda;
        scalar_t *C = pPrm->C;
        int ldc = pPrm->ldc;
        unsigned M = pPrm->M;
        do {
          fp_vector_t a = MM_BROADCAST_Sx(A); A += lda;
          MM_MASKSTOREU_Px(C, mask_n, MM_MUL_Px(b, a)); C += ldc;
        } while (--M);
      }
    } else {
      fp_vector_t alpha = MM_BROADCAST_Sx(&pPrm->alpha);
      unsigned nDiv = pPrm->N / (SIMD_FACTOR*8);
      while (nDiv != 0) {
        const scalar_t *B = pPrm->B;
        fp_vector_t b0 = MM_LOADU_Px(B); B += SIMD_FACTOR;
        fp_vector_t b1 = MM_LOADU_Px(B); B += SIMD_FACTOR;
        fp_vector_t b2 = MM_LOADU_Px(B); B += SIMD_FACTOR;
        fp_vector_t b3 = MM_LOADU_Px(B); B += SIMD_FACTOR;
        fp_vector_t b4 = MM_LOADU_Px(B); B += SIMD_FACTOR;
        fp_vector_t b5 = MM_LOADU_Px(B); B += SIMD_FACTOR;
        fp_vector_t b6 = MM_LOADU_Px(B); B += SIMD_FACTOR;
        fp_vector_t b7 = MM_LOADU_Px(B); B += SIMD_FACTOR;

        const scalar_t *A = pPrm->A;
        int lda = pPrm->lda;
        scalar_t *C = pPrm->C;
        int ldc = pPrm->ldc;
        unsigned M = pPrm->M;
        do {
          fp_vector_t a = MM_BROADCAST_Sx(A); A += lda;
          a = MM_MUL_Px(a, alpha);
          MM_STOREU_Px(C+SIMD_FACTOR*0, MM_MUL_Px(b0, a));
          MM_STOREU_Px(C+SIMD_FACTOR*1, MM_MUL_Px(b1, a));
          MM_STOREU_Px(C+SIMD_FACTOR*2, MM_MUL_Px(b2, a));
          MM_STOREU_Px(C+SIMD_FACTOR*3, MM_MUL_Px(b3, a));
          MM_STOREU_Px(C+SIMD_FACTOR*4, MM_MUL_Px(b4, a));
          MM_STOREU_Px(C+SIMD_FACTOR*5, MM_MUL_Px(b5, a));
          MM_STOREU_Px(C+SIMD_FACTOR*6, MM_MUL_Px(b6, a));
          MM_STOREU_Px(C+SIMD_FACTOR*7, MM_MUL_Px(b7, a));
          C += ldc;
        } while (--M);

        pPrm->B += SIMD_FACTOR*8;
        pPrm->C += SIMD_FACTOR*8;
        --nDiv;
      }

      unsigned nRem = pPrm->N % (SIMD_FACTOR*8);
      nDiv = nRem / SIMD_FACTOR;
      const scalar_t *B = pPrm->B;
      while (nDiv != 0) {
        fp_vector_t b = MM_LOADU_Px(B); B += SIMD_FACTOR;
        b = MM_MUL_Px(b, alpha);

        const scalar_t *A = pPrm->A;
        int lda = pPrm->lda;
        scalar_t *C = pPrm->C;
        int ldc = pPrm->ldc;
        unsigned M = pPrm->M;
        do {
          fp_vector_t a = MM_BROADCAST_Sx(A); A += lda;
          MM_STOREU_Px(C, MM_MUL_Px(b, a));   C += ldc;
        } while (--M);

        pPrm->C += SIMD_FACTOR;
        --nDiv;
      }

      if (pPrm->N % SIMD_FACTOR) {
        int_vector_t mask_n = pPrm->mask_n;
        fp_vector_t b = MM_MASKLOADU_Px(B, mask_n);
        b = MM_MUL_Px(b, alpha);

        const scalar_t *A = pPrm->A;
        int lda = pPrm->lda;
        scalar_t *C = pPrm->C;
        int ldc = pPrm->ldc;
        unsigned M = pPrm->M;
        do {
          fp_vector_t a = MM_BROADCAST_Sx(A); A += lda;
          MM_MASKSTOREU_Px(C, mask_n, MM_MUL_Px(b, a)); C += ldc;
        } while (--M);
      }
    }
  } else {
    fp_vector_t beta = MM_BROADCAST_Sx(&pPrm->beta);
    if (pPrm->alpha == 1.0f) {
      unsigned nDiv = pPrm->N / (SIMD_FACTOR*8);
      while (nDiv != 0) {
        const scalar_t *B = pPrm->B;
        fp_vector_t b0 = MM_LOADU_Px(B); B += SIMD_FACTOR;
        fp_vector_t b1 = MM_LOADU_Px(B); B += SIMD_FACTOR;
        fp_vector_t b2 = MM_LOADU_Px(B); B += SIMD_FACTOR;
        fp_vector_t b3 = MM_LOADU_Px(B); B += SIMD_FACTOR;
        fp_vector_t b4 = MM_LOADU_Px(B); B += SIMD_FACTOR;
        fp_vector_t b5 = MM_LOADU_Px(B); B += SIMD_FACTOR;
        fp_vector_t b6 = MM_LOADU_Px(B); B += SIMD_FACTOR;
        fp_vector_t b7 = MM_LOADU_Px(B); B += SIMD_FACTOR;

        const scalar_t *A = pPrm->A;
        int lda = pPrm->lda;
        scalar_t *C = pPrm->C;
        int ldc = pPrm->ldc;
        unsigned M = pPrm->M;
        do {
          fp_vector_t a = MM_BROADCAST_Sx(A); A += lda;
          MM_STOREU_Px(C+SIMD_FACTOR*0, MM_FMADD(b0, a, MM_MUL_Px(beta, MM_LOADU_Px(C+SIMD_FACTOR*0))));
          MM_STOREU_Px(C+SIMD_FACTOR*1, MM_FMADD(b1, a, MM_MUL_Px(beta, MM_LOADU_Px(C+SIMD_FACTOR*1))));
          MM_STOREU_Px(C+SIMD_FACTOR*2, MM_FMADD(b2, a, MM_MUL_Px(beta, MM_LOADU_Px(C+SIMD_FACTOR*2))));
          MM_STOREU_Px(C+SIMD_FACTOR*3, MM_FMADD(b3, a, MM_MUL_Px(beta, MM_LOADU_Px(C+SIMD_FACTOR*3))));
          MM_STOREU_Px(C+SIMD_FACTOR*4, MM_FMADD(b4, a, MM_MUL_Px(beta, MM_LOADU_Px(C+SIMD_FACTOR*4))));
          MM_STOREU_Px(C+SIMD_FACTOR*5, MM_FMADD(b5, a, MM_MUL_Px(beta, MM_LOADU_Px(C+SIMD_FACTOR*5))));
          MM_STOREU_Px(C+SIMD_FACTOR*6, MM_FMADD(b6, a, MM_MUL_Px(beta, MM_LOADU_Px(C+SIMD_FACTOR*6))));
          MM_STOREU_Px(C+SIMD_FACTOR*7, MM_FMADD(b7, a, MM_MUL_Px(beta, MM_LOADU_Px(C+SIMD_FACTOR*7))));
          C += ldc;
        } while (--M);

        pPrm->B += SIMD_FACTOR*8;
        pPrm->C += SIMD_FACTOR*8;
        --nDiv;
      }

      unsigned nRem = pPrm->N % (SIMD_FACTOR*8);
      nDiv = nRem / SIMD_FACTOR;
      const scalar_t *B = pPrm->B;
      while (nDiv != 0) {
        fp_vector_t b = MM_LOADU_Px(B); B += SIMD_FACTOR;

        const scalar_t *A = pPrm->A;
        int lda = pPrm->lda;
        scalar_t *C = pPrm->C;
        int ldc = pPrm->ldc;
        unsigned M = pPrm->M;
        do {
          fp_vector_t a = MM_BROADCAST_Sx(A); A += lda;
          fp_vector_t c = MM_LOADU_Px(C);
          MM_STOREU_Px(C, MM_FMADD(b, a, MM_MUL_Px(beta, c))); C += ldc;
        } while (--M);

        pPrm->C += SIMD_FACTOR;
        --nDiv;
      }

      if (pPrm->N % SIMD_FACTOR) {
        int_vector_t mask_n = pPrm->mask_n;
        fp_vector_t b = MM_MASKLOADU_Px(B, mask_n);

        const scalar_t *A = pPrm->A;
        int lda = pPrm->lda;
        scalar_t *C = pPrm->C;
        int ldc = pPrm->ldc;
        unsigned M = pPrm->M;
        do {
          fp_vector_t a = MM_BROADCAST_Sx(A); A += lda;
          fp_vector_t c = MM_MASKLOADU_Px(C, mask_n);
          MM_MASKSTOREU_Px(C, mask_n, MM_FMADD(b, a, MM_MUL_Px(beta, c))); C += ldc;
        } while (--M);
      }
    } else {
      fp_vector_t alpha = MM_BROADCAST_Sx(&pPrm->alpha);
      unsigned nDiv = pPrm->N / (SIMD_FACTOR*8);
      while (nDiv != 0) {
        const scalar_t *B = pPrm->B;
        fp_vector_t b0 = MM_LOADU_Px(B); B += SIMD_FACTOR;
        fp_vector_t b1 = MM_LOADU_Px(B); B += SIMD_FACTOR;
        fp_vector_t b2 = MM_LOADU_Px(B); B += SIMD_FACTOR;
        fp_vector_t b3 = MM_LOADU_Px(B); B += SIMD_FACTOR;
        fp_vector_t b4 = MM_LOADU_Px(B); B += SIMD_FACTOR;
        fp_vector_t b5 = MM_LOADU_Px(B); B += SIMD_FACTOR;
        fp_vector_t b6 = MM_LOADU_Px(B); B += SIMD_FACTOR;
        fp_vector_t b7 = MM_LOADU_Px(B); B += SIMD_FACTOR;

        const scalar_t *A = pPrm->A;
        int lda = pPrm->lda;
        scalar_t *C = pPrm->C;
        int ldc = pPrm->ldc;
        unsigned M = pPrm->M;
        do {
          fp_vector_t a = MM_BROADCAST_Sx(A); A += lda;
          a = MM_MUL_Px(a, alpha);
          MM_STOREU_Px(C+SIMD_FACTOR*0, MM_FMADD(b0, a, MM_MUL_Px(beta, MM_LOADU_Px(C+SIMD_FACTOR*0))));
          MM_STOREU_Px(C+SIMD_FACTOR*1, MM_FMADD(b1, a, MM_MUL_Px(beta, MM_LOADU_Px(C+SIMD_FACTOR*1))));
          MM_STOREU_Px(C+SIMD_FACTOR*2, MM_FMADD(b2, a, MM_MUL_Px(beta, MM_LOADU_Px(C+SIMD_FACTOR*2))));
          MM_STOREU_Px(C+SIMD_FACTOR*3, MM_FMADD(b3, a, MM_MUL_Px(beta, MM_LOADU_Px(C+SIMD_FACTOR*3))));
          MM_STOREU_Px(C+SIMD_FACTOR*4, MM_FMADD(b4, a, MM_MUL_Px(beta, MM_LOADU_Px(C+SIMD_FACTOR*4))));
          MM_STOREU_Px(C+SIMD_FACTOR*5, MM_FMADD(b5, a, MM_MUL_Px(beta, MM_LOADU_Px(C+SIMD_FACTOR*5))));
          MM_STOREU_Px(C+SIMD_FACTOR*6, MM_FMADD(b6, a, MM_MUL_Px(beta, MM_LOADU_Px(C+SIMD_FACTOR*6))));
          MM_STOREU_Px(C+SIMD_FACTOR*7, MM_FMADD(b7, a, MM_MUL_Px(beta, MM_LOADU_Px(C+SIMD_FACTOR*7))));
          C += ldc;
        } while (--M);

        pPrm->B += SIMD_FACTOR*8;
        pPrm->C += SIMD_FACTOR*8;
        --nDiv;
      }

      unsigned nRem = pPrm->N % (SIMD_FACTOR*8);
      nDiv = nRem / SIMD_FACTOR;
      const scalar_t *B = pPrm->B;
      while (nDiv != 0) {
        fp_vector_t b = MM_LOADU_Px(B); B += SIMD_FACTOR;
        b = MM_MUL_Px(b, alpha);

        const scalar_t *A = pPrm->A;
        int lda = pPrm->lda;
        scalar_t *C = pPrm->C;
        int ldc = pPrm->ldc;
        unsigned M = pPrm->M;
        do {
          fp_vector_t a = MM_BROADCAST_Sx(A); A += lda;
          fp_vector_t c = MM_LOADU_Px(C);
          MM_STOREU_Px(C, MM_FMADD(b, a, MM_MUL_Px(beta, c))); C += ldc;
        } while (--M);

        pPrm->C += SIMD_FACTOR;
        --nDiv;
      }

      if (pPrm->N % SIMD_FACTOR) {
        int_vector_t mask_n = pPrm->mask_n;
        fp_vector_t b = MM_MASKLOADU_Px(B, mask_n);
        b = MM_MUL_Px(b, alpha);

        const scalar_t *A = pPrm->A;
        int lda = pPrm->lda;
        scalar_t *C = pPrm->C;
        int ldc = pPrm->ldc;
        unsigned M = pPrm->M;
        do {
          fp_vector_t a = MM_BROADCAST_Sx(A); A += lda;
          fp_vector_t c = MM_MASKLOADU_Px(C, mask_n);
          MM_MASKSTOREU_Px(C, mask_n, MM_FMADD(b, a, MM_MUL_Px(beta, c))); C += ldc;
        } while (--M);
      }
    }
  }
}

static void noncblas_sgemm_k2(noncblas_sgemm_prm_t* pPrm)
{
  if (pPrm->beta == 0) {
    if (pPrm->alpha == 1.0f) {
      unsigned nDiv = pPrm->N / (SIMD_FACTOR*4);
      while (nDiv != 0) {
        const scalar_t *B0 = pPrm->B;
        const scalar_t *B1 = B0 + pPrm->ldb;
        fp_vector_t b00 = MM_LOADU_Px(B0); B0 += SIMD_FACTOR;
        fp_vector_t b01 = MM_LOADU_Px(B0); B0 += SIMD_FACTOR;
        fp_vector_t b02 = MM_LOADU_Px(B0); B0 += SIMD_FACTOR;
        fp_vector_t b03 = MM_LOADU_Px(B0); B0 += SIMD_FACTOR;
        fp_vector_t b10 = MM_LOADU_Px(B1); B1 += SIMD_FACTOR;
        fp_vector_t b11 = MM_LOADU_Px(B1); B1 += SIMD_FACTOR;
        fp_vector_t b12 = MM_LOADU_Px(B1); B1 += SIMD_FACTOR;
        fp_vector_t b13 = MM_LOADU_Px(B1); B1 += SIMD_FACTOR;

        const scalar_t *A = pPrm->A;
        int lda = pPrm->lda;
        scalar_t *C = pPrm->C;
        int ldc = pPrm->ldc;
        unsigned M = pPrm->M;
        do {
          fp_vector_t a0 = MM_BROADCAST_Sx(A+0);
          fp_vector_t a1 = MM_BROADCAST_Sx(A+1);
          A += lda;
          fp_vector_t acc0 = MM_MUL_Px(b00, a0);
          fp_vector_t acc1 = MM_MUL_Px(b01, a0);
          fp_vector_t acc2 = MM_MUL_Px(b02, a0);
          fp_vector_t acc3 = MM_MUL_Px(b03, a0);
          acc0 = MM_FMADD(b10, a1, acc0);
          acc1 = MM_FMADD(b11, a1, acc1);
          acc2 = MM_FMADD(b12, a1, acc2);
          acc3 = MM_FMADD(b13, a1, acc3);
          MM_STOREU_Px(C+SIMD_FACTOR*0, acc0);
          MM_STOREU_Px(C+SIMD_FACTOR*1, acc1);
          MM_STOREU_Px(C+SIMD_FACTOR*2, acc2);
          MM_STOREU_Px(C+SIMD_FACTOR*3, acc3);
          C += ldc;
        } while (--M);

        pPrm->B += SIMD_FACTOR*4;
        pPrm->C += SIMD_FACTOR*4;
        --nDiv;
      }

      unsigned nRem = pPrm->N % (SIMD_FACTOR*4);
      nDiv = nRem / SIMD_FACTOR;
      const scalar_t *B = pPrm->B;
      int ldb = pPrm->ldb;
      while (nDiv != 0) {
        fp_vector_t b0 = MM_LOADU_Px(B);
        fp_vector_t b1 = MM_LOADU_Px(B+ldb);
        B += SIMD_FACTOR;

        const scalar_t *A = pPrm->A;
        int lda = pPrm->lda;
        scalar_t *C = pPrm->C;
        int ldc = pPrm->ldc;
        unsigned M = pPrm->M;
        do {
          fp_vector_t a0 = MM_BROADCAST_Sx(A+0);
          fp_vector_t a1 = MM_BROADCAST_Sx(A+1);
          A += lda;
          fp_vector_t acc = MM_MUL_Px(b0, a0);
          acc = MM_FMADD(b1, a1, acc);
          MM_STOREU_Px(C, acc);   C += ldc;
        } while (--M);

        pPrm->C += SIMD_FACTOR;
        --nDiv;
      }

      if (pPrm->N % SIMD_FACTOR) {
        int_vector_t mask_n = pPrm->mask_n;
        fp_vector_t b0 = MM_MASKLOADU_Px(B,     mask_n);
        fp_vector_t b1 = MM_MASKLOADU_Px(B+ldb, mask_n);

        const scalar_t *A = pPrm->A;
        int lda = pPrm->lda;
        scalar_t *C = pPrm->C;
        int ldc = pPrm->ldc;
        unsigned M = pPrm->M;
        do {
          fp_vector_t a0 = MM_BROADCAST_Sx(A+0);
          fp_vector_t a1 = MM_BROADCAST_Sx(A+1);
          A += lda;
          fp_vector_t acc = MM_MUL_Px(b0, a0);
          acc = MM_FMADD(b1, a1, acc);
          MM_MASKSTOREU_Px(C, mask_n, acc); C += ldc;
        } while (--M);
      }
    } else {
      fp_vector_t alpha = MM_BROADCAST_Sx(&pPrm->alpha);
      unsigned nDiv = pPrm->N / (SIMD_FACTOR*4);
      while (nDiv != 0) {
        const scalar_t *B0 = pPrm->B;
        const scalar_t *B1 = B0 + pPrm->ldb;
        fp_vector_t b00 = MM_LOADU_Px(B0); B0 += SIMD_FACTOR;
        fp_vector_t b01 = MM_LOADU_Px(B0); B0 += SIMD_FACTOR;
        fp_vector_t b02 = MM_LOADU_Px(B0); B0 += SIMD_FACTOR;
        fp_vector_t b03 = MM_LOADU_Px(B0); B0 += SIMD_FACTOR;
        fp_vector_t b10 = MM_LOADU_Px(B1); B1 += SIMD_FACTOR;
        fp_vector_t b11 = MM_LOADU_Px(B1); B1 += SIMD_FACTOR;
        fp_vector_t b12 = MM_LOADU_Px(B1); B1 += SIMD_FACTOR;
        fp_vector_t b13 = MM_LOADU_Px(B1); B1 += SIMD_FACTOR;

        b00 = MM_MUL_Px(b00, alpha);
        b01 = MM_MUL_Px(b01, alpha);
        b02 = MM_MUL_Px(b02, alpha);
        b03 = MM_MUL_Px(b03, alpha);
        b10 = MM_MUL_Px(b10, alpha);
        b11 = MM_MUL_Px(b11, alpha);
        b12 = MM_MUL_Px(b12, alpha);
        b13 = MM_MUL_Px(b13, alpha);

        const scalar_t *A = pPrm->A;
        int lda = pPrm->lda;
        scalar_t *C = pPrm->C;
        int ldc = pPrm->ldc;
        unsigned M = pPrm->M;
        do {
          fp_vector_t a0 = MM_BROADCAST_Sx(A+0);
          fp_vector_t a1 = MM_BROADCAST_Sx(A+1);
          A += lda;
          fp_vector_t acc0 = MM_MUL_Px(b00, a0);
          fp_vector_t acc1 = MM_MUL_Px(b01, a0);
          fp_vector_t acc2 = MM_MUL_Px(b02, a0);
          fp_vector_t acc3 = MM_MUL_Px(b03, a0);
          acc0 = MM_FMADD(b10, a1, acc0);
          acc1 = MM_FMADD(b11, a1, acc1);
          acc2 = MM_FMADD(b12, a1, acc2);
          acc3 = MM_FMADD(b13, a1, acc3);
          MM_STOREU_Px(C+SIMD_FACTOR*0, acc0);
          MM_STOREU_Px(C+SIMD_FACTOR*1, acc1);
          MM_STOREU_Px(C+SIMD_FACTOR*2, acc2);
          MM_STOREU_Px(C+SIMD_FACTOR*3, acc3);
          C += ldc;
        } while (--M);

        pPrm->B += SIMD_FACTOR*4;
        pPrm->C += SIMD_FACTOR*4;
        --nDiv;
      }

      unsigned nRem = pPrm->N % (SIMD_FACTOR*4);
      nDiv = nRem / SIMD_FACTOR;
      const scalar_t *B = pPrm->B;
      int ldb = pPrm->ldb;
      while (nDiv != 0) {
        fp_vector_t b0 = MM_LOADU_Px(B);
        fp_vector_t b1 = MM_LOADU_Px(B+ldb);
        B += SIMD_FACTOR;

        b0 = MM_MUL_Px(b0, alpha);
        b1 = MM_MUL_Px(b1, alpha);

        const scalar_t *A = pPrm->A;
        int lda = pPrm->lda;
        scalar_t *C = pPrm->C;
        int ldc = pPrm->ldc;
        unsigned M = pPrm->M;
        do {
          fp_vector_t a0 = MM_BROADCAST_Sx(A+0);
          fp_vector_t a1 = MM_BROADCAST_Sx(A+1);
          A += lda;
          fp_vector_t acc = MM_MUL_Px(b0, a0);
          acc = MM_FMADD(b1, a1, acc);
          MM_STOREU_Px(C, acc);   C += ldc;
        } while (--M);

        pPrm->C += SIMD_FACTOR;
        --nDiv;
      }

      if (pPrm->N % SIMD_FACTOR) {
        int_vector_t mask_n = pPrm->mask_n;
        fp_vector_t b0 = MM_MASKLOADU_Px(B,     mask_n);
        fp_vector_t b1 = MM_MASKLOADU_Px(B+ldb, mask_n);

        b0 = MM_MUL_Px(b0, alpha);
        b1 = MM_MUL_Px(b1, alpha);

        const scalar_t *A = pPrm->A;
        int lda = pPrm->lda;
        scalar_t *C = pPrm->C;
        int ldc = pPrm->ldc;
        unsigned M = pPrm->M;
        do {
          fp_vector_t a0 = MM_BROADCAST_Sx(A+0);
          fp_vector_t a1 = MM_BROADCAST_Sx(A+1);
          A += lda;
          fp_vector_t acc = MM_MUL_Px(b0, a0);
          acc = MM_FMADD(b1, a1, acc);
          MM_MASKSTOREU_Px(C, mask_n, acc); C += ldc;
        } while (--M);
      }
    }
  } else {
    fp_vector_t beta = MM_BROADCAST_Sx(&pPrm->beta);
    if (pPrm->alpha == 1.0f) {
      unsigned nDiv = pPrm->N / (SIMD_FACTOR*4);
      while (nDiv != 0) {
        const scalar_t *B0 = pPrm->B;
        const scalar_t *B1 = B0 + pPrm->ldb;
        fp_vector_t b00 = MM_LOADU_Px(B0); B0 += SIMD_FACTOR;
        fp_vector_t b01 = MM_LOADU_Px(B0); B0 += SIMD_FACTOR;
        fp_vector_t b02 = MM_LOADU_Px(B0); B0 += SIMD_FACTOR;
        fp_vector_t b03 = MM_LOADU_Px(B0); B0 += SIMD_FACTOR;
        fp_vector_t b10 = MM_LOADU_Px(B1); B1 += SIMD_FACTOR;
        fp_vector_t b11 = MM_LOADU_Px(B1); B1 += SIMD_FACTOR;
        fp_vector_t b12 = MM_LOADU_Px(B1); B1 += SIMD_FACTOR;
        fp_vector_t b13 = MM_LOADU_Px(B1); B1 += SIMD_FACTOR;

        const scalar_t *A = pPrm->A;
        int lda = pPrm->lda;
        scalar_t *C = pPrm->C;
        int ldc = pPrm->ldc;
        unsigned M = pPrm->M;
        do {
          fp_vector_t a0 = MM_BROADCAST_Sx(A+0);
          fp_vector_t a1 = MM_BROADCAST_Sx(A+1);
          A += lda;
          fp_vector_t acc;

          acc = MM_MUL_Px(MM_LOADU_Px(C+SIMD_FACTOR*0), beta);
          acc = MM_FMADD(b00, a0, acc);
          acc = MM_FMADD(b10, a1, acc);
          MM_STOREU_Px(C+SIMD_FACTOR*0, acc);

          acc = MM_MUL_Px(MM_LOADU_Px(C+SIMD_FACTOR*1), beta);
          acc = MM_FMADD(b01, a0, acc);
          acc = MM_FMADD(b11, a1, acc);
          MM_STOREU_Px(C+SIMD_FACTOR*1, acc);

          acc = MM_MUL_Px(MM_LOADU_Px(C+SIMD_FACTOR*2), beta);
          acc = MM_FMADD(b02, a0, acc);
          acc = MM_FMADD(b12, a1, acc);
          MM_STOREU_Px(C+SIMD_FACTOR*2, acc);

          acc = MM_MUL_Px(MM_LOADU_Px(C+SIMD_FACTOR*3), beta);
          acc = MM_FMADD(b03, a0, acc);
          acc = MM_FMADD(b13, a1, acc);
          MM_STOREU_Px(C+SIMD_FACTOR*3, acc);

          C += ldc;
        } while (--M);

        pPrm->B += SIMD_FACTOR*4;
        pPrm->C += SIMD_FACTOR*4;
        --nDiv;
      }

      unsigned nRem = pPrm->N % (SIMD_FACTOR*4);
      nDiv = nRem / SIMD_FACTOR;
      const scalar_t *B = pPrm->B;
      int ldb = pPrm->ldb;
      while (nDiv != 0) {
        fp_vector_t b0 = MM_LOADU_Px(B);
        fp_vector_t b1 = MM_LOADU_Px(B+ldb);
        B += SIMD_FACTOR;

        const scalar_t *A = pPrm->A;
        int lda = pPrm->lda;
        scalar_t *C = pPrm->C;
        int ldc = pPrm->ldc;
        unsigned M = pPrm->M;
        do {
          fp_vector_t a0 = MM_BROADCAST_Sx(A+0);
          fp_vector_t a1 = MM_BROADCAST_Sx(A+1);
          A += lda;
          fp_vector_t c = MM_LOADU_Px(C);
          fp_vector_t acc = MM_MUL_Px(beta, c);
          acc = MM_FMADD(b0, a0, acc);
          acc = MM_FMADD(b1, a1, acc);
          MM_STOREU_Px(C, acc); C += ldc;
        } while (--M);

        pPrm->C += SIMD_FACTOR;
        --nDiv;
      }

      if (pPrm->N % SIMD_FACTOR) {
        int_vector_t mask_n = pPrm->mask_n;
        fp_vector_t b0 = MM_MASKLOADU_Px(B,     mask_n);
        fp_vector_t b1 = MM_MASKLOADU_Px(B+ldb, mask_n);

        const scalar_t *A = pPrm->A;
        int lda = pPrm->lda;
        scalar_t *C = pPrm->C;
        int ldc = pPrm->ldc;
        unsigned M = pPrm->M;
        do {
          fp_vector_t a0 = MM_BROADCAST_Sx(A+0);
          fp_vector_t a1 = MM_BROADCAST_Sx(A+1);
          A += lda;
          fp_vector_t c = MM_MASKLOADU_Px(C, mask_n);
          fp_vector_t acc = MM_MUL_Px(beta, c);
          acc = MM_FMADD(b0, a0, acc);
          acc = MM_FMADD(b1, a1, acc);
          MM_MASKSTOREU_Px(C, mask_n, acc); C += ldc;
        } while (--M);
      }
    } else {
      fp_vector_t alpha = MM_BROADCAST_Sx(&pPrm->alpha);
      unsigned nDiv = pPrm->N / (SIMD_FACTOR*4);
      while (nDiv != 0) {
        const scalar_t *B0 = pPrm->B;
        const scalar_t *B1 = B0 + pPrm->ldb;
        fp_vector_t b00 = MM_LOADU_Px(B0); B0 += SIMD_FACTOR;
        fp_vector_t b01 = MM_LOADU_Px(B0); B0 += SIMD_FACTOR;
        fp_vector_t b02 = MM_LOADU_Px(B0); B0 += SIMD_FACTOR;
        fp_vector_t b03 = MM_LOADU_Px(B0); B0 += SIMD_FACTOR;
        fp_vector_t b10 = MM_LOADU_Px(B1); B1 += SIMD_FACTOR;
        fp_vector_t b11 = MM_LOADU_Px(B1); B1 += SIMD_FACTOR;
        fp_vector_t b12 = MM_LOADU_Px(B1); B1 += SIMD_FACTOR;
        fp_vector_t b13 = MM_LOADU_Px(B1); B1 += SIMD_FACTOR;

        b00 = MM_MUL_Px(b00, alpha);
        b01 = MM_MUL_Px(b01, alpha);
        b02 = MM_MUL_Px(b02, alpha);
        b03 = MM_MUL_Px(b03, alpha);
        b10 = MM_MUL_Px(b10, alpha);
        b11 = MM_MUL_Px(b11, alpha);
        b12 = MM_MUL_Px(b12, alpha);
        b13 = MM_MUL_Px(b13, alpha);

        const scalar_t *A = pPrm->A;
        int lda = pPrm->lda;
        scalar_t *C = pPrm->C;
        int ldc = pPrm->ldc;
        unsigned M = pPrm->M;
        do {
          fp_vector_t a0 = MM_BROADCAST_Sx(A+0);
          fp_vector_t a1 = MM_BROADCAST_Sx(A+1);
          A += lda;
          fp_vector_t acc;

          acc = MM_MUL_Px(MM_LOADU_Px(C+SIMD_FACTOR*0), beta);
          acc = MM_FMADD(b00, a0, acc);
          acc = MM_FMADD(b10, a1, acc);
          MM_STOREU_Px(C+SIMD_FACTOR*0, acc);

          acc = MM_MUL_Px(MM_LOADU_Px(C+SIMD_FACTOR*1), beta);
          acc = MM_FMADD(b01, a0, acc);
          acc = MM_FMADD(b11, a1, acc);
          MM_STOREU_Px(C+SIMD_FACTOR*1, acc);

          acc = MM_MUL_Px(MM_LOADU_Px(C+SIMD_FACTOR*2), beta);
          acc = MM_FMADD(b02, a0, acc);
          acc = MM_FMADD(b12, a1, acc);
          MM_STOREU_Px(C+SIMD_FACTOR*2, acc);

          acc = MM_MUL_Px(MM_LOADU_Px(C+SIMD_FACTOR*3), beta);
          acc = MM_FMADD(b03, a0, acc);
          acc = MM_FMADD(b13, a1, acc);
          MM_STOREU_Px(C+SIMD_FACTOR*3, acc);

          C += ldc;
        } while (--M);

        pPrm->B += SIMD_FACTOR*4;
        pPrm->C += SIMD_FACTOR*4;
        --nDiv;
      }

      unsigned nRem = pPrm->N % (SIMD_FACTOR*4);
      nDiv = nRem / SIMD_FACTOR;
      const scalar_t *B = pPrm->B;
      int ldb = pPrm->ldb;
      while (nDiv != 0) {
        fp_vector_t b0 = MM_LOADU_Px(B);
        fp_vector_t b1 = MM_LOADU_Px(B+ldb);
        B += SIMD_FACTOR;
        b0 = MM_MUL_Px(b0, alpha);
        b1 = MM_MUL_Px(b1, alpha);

        const scalar_t *A = pPrm->A;
        int lda = pPrm->lda;
        scalar_t *C = pPrm->C;
        int ldc = pPrm->ldc;
        unsigned M = pPrm->M;
        do {
          fp_vector_t a0 = MM_BROADCAST_Sx(A+0);
          fp_vector_t a1 = MM_BROADCAST_Sx(A+1);
          A += lda;
          fp_vector_t c = MM_LOADU_Px(C);
          fp_vector_t acc = MM_MUL_Px(beta, c);
          acc = MM_FMADD(b0, a0, acc);
          acc = MM_FMADD(b1, a1, acc);
          MM_STOREU_Px(C, acc); C += ldc;
        } while (--M);

        pPrm->C += SIMD_FACTOR;
        --nDiv;
      }

      if (pPrm->N % SIMD_FACTOR) {
        int_vector_t mask_n = pPrm->mask_n;
        fp_vector_t b0 = MM_MASKLOADU_Px(B,     mask_n);
        fp_vector_t b1 = MM_MASKLOADU_Px(B+ldb, mask_n);
        b0 = MM_MUL_Px(b0, alpha);
        b1 = MM_MUL_Px(b1, alpha);

        const scalar_t *A = pPrm->A;
        int lda = pPrm->lda;
        scalar_t *C = pPrm->C;
        int ldc = pPrm->ldc;
        unsigned M = pPrm->M;
        do {
          fp_vector_t a0 = MM_BROADCAST_Sx(A+0);
          fp_vector_t a1 = MM_BROADCAST_Sx(A+1);
          A += lda;
          fp_vector_t c = MM_MASKLOADU_Px(C, mask_n);
          fp_vector_t acc = MM_MUL_Px(beta, c);
          acc = MM_FMADD(b0, a0, acc);
          acc = MM_FMADD(b1, a1, acc);
          MM_MASKSTOREU_Px(C, mask_n, acc); C += ldc;
        } while (--M);
      }
    }
  }
}

static void noncblas_sgemm_k3(noncblas_sgemm_prm_t* pPrm)
{
}

static void noncblas_sgemm_k4(noncblas_sgemm_prm_t* pPrm)
{
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

  noncblas_sgemm_prm_t prm={0};
  unsigned NRem = ((unsigned)(N-1) % SIMD_FACTOR) + 1;
  // mask on elements of rightmost SIMD word in B and C
  for (unsigned i = 0; i < NRem; ++i)
    ((uint8_t*)&prm.mask_n)[sizeof(scalar_t)*(i+1)-1] = (uint8_t)(-1);

  prm.A     = A;
  prm.B     = B;
  prm.C     = C;
  prm.M     = M;
  prm.N     = N;
  prm.K     = K;
  prm.lda   = lda;
  prm.ldb   = ldb;
  prm.ldc   = ldc;
  prm.alpha = alpha;
  prm.beta  = beta;

  if (K < 5) {
    switch (K) {
      case 1:  noncblas_sgemm_k1(&prm); break;

      case 2:  noncblas_sgemm_k2(&prm); break;

      case 3:  noncblas_sgemm_k3(&prm); break;

      case 4:  noncblas_sgemm_k4(&prm); break;

      default:
        break;
    }
  } else if (N > SIMD_FACTOR) {
    noncblas_sgemm_wide_n(&prm);
  } else if (N >= 1) {
    noncblas_sgemm_narrow_n(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }
}

#ifdef  NONCBLAS_SGEMM_TUNE
void tune_name(int m_step, int k_step) {
  //st_m_step = m_step;
  st_k_step = k_step;
}
#endif