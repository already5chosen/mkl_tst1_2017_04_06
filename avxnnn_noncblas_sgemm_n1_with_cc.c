enum {
 SIMD_FACTOR          = sizeof(fp_vector_t)/sizeof(scalar_t),
 A_WORDS_PER_ITER     = 5,
 B_WORDS_PER_ITER     = 2,
 SIMD_ELEM_PEC_COL_MJ = B_WORDS_PER_ITER,
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
  int           masked_b;
  scalar_t      alpha;
  scalar_t      beta;
  int_vector4_t mask_a[2]; // [0]= all 1s, [2] = last vector4 word
  int_vector_t  mask_b[3]; // [0]= all 1s, [1] = all ones or last SIMD word, [2] = last SIMD word
  fp_vector_t*  bb; // [SIMD_ELEM_PEC_COL_MJ*k_step];
  fp_vector4_t* aa; // [(k_step*m_step_max)/4];
  scalar_t*     cc;
} noncblas_sgemm_prm_t;

// major core - inner loop processes 2 SIMD columns of B x 5 rows of A
static void fma256_noncblas_sgemm_core_mj(
 const noncblas_sgemm_prm_t* pPrm,
 scalar_t*                   C,
 int                         nRows)    // 0 < nRows    <= k_step
{
  int ldc = pPrm->ldc;
  int kSteps = (unsigned)(nRows-1) / 4 + 1;
  int m;
  const fp_vector4_t* A = pPrm->aa;
  const ptrdiff_t preftechDistance = ldc*A_WORDS_PER_ITER*sizeof(*C);
  for (m = 0; m < pPrm->M-A_WORDS_PER_ITER+1;
    A += kSteps*A_WORDS_PER_ITER,
    m += A_WORDS_PER_ITER) {
    const fp_vector_t* Bcol = pPrm->bb;
    const scalar_t*    ARow = (const scalar_t*)(A);
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
    #define Prefetch2lines(x) \
      _mm_prefetch((char*)&((x)[SIMD_FACTOR*0])+preftechDistance, _MM_HINT_T0); \
      _mm_prefetch((char*)&((x)[SIMD_FACTOR*1])+preftechDistance, _MM_HINT_T0);

    if (!pPrm->masked_b) {
      if (pPrm->c_option == C_OPTION_UPDATE) {
        #define UPDATE_CCOL(ccol, acc0, acc1) \
        MM_STOREU_Px(&((ccol)[SIMD_FACTOR*0]), MM_FMADD((acc0), alpha_ps, MM_LOADU_Px(&((ccol)[SIMD_FACTOR*0])))); \
        MM_STOREU_Px(&((ccol)[SIMD_FACTOR*1]), MM_FMADD((acc1), alpha_ps, MM_LOADU_Px(&((ccol)[SIMD_FACTOR*1]))));

        UPDATE_CCOL(C, acc00, acc10);  Prefetch2lines(C); C += ldc;
        UPDATE_CCOL(C, acc01, acc11);  Prefetch2lines(C); C += ldc;
        UPDATE_CCOL(C, acc02, acc12);  Prefetch2lines(C); C += ldc;
        UPDATE_CCOL(C, acc03, acc13);  Prefetch2lines(C); C += ldc;
        UPDATE_CCOL(C, acc04, acc14);  Prefetch2lines(C); C += ldc;

        #undef UPDATE_CCOL
      } else if (pPrm->c_option == C_OPTION_REPLACE) {
        #define UPDATE_CCOL(ccol, acc0, acc1) \
        MM_STOREU_Px(&((ccol)[SIMD_FACTOR*0]), MM_MUL_Px((acc0), alpha_ps)); \
        MM_STOREU_Px(&((ccol)[SIMD_FACTOR*1]), MM_MUL_Px((acc1), alpha_ps));

        UPDATE_CCOL(C, acc00, acc10);  Prefetch2lines(C); C += ldc;
        UPDATE_CCOL(C, acc01, acc11);  Prefetch2lines(C); C += ldc;
        UPDATE_CCOL(C, acc02, acc12);  Prefetch2lines(C); C += ldc;
        UPDATE_CCOL(C, acc03, acc13);  Prefetch2lines(C); C += ldc;
        UPDATE_CCOL(C, acc04, acc14);  Prefetch2lines(C); C += ldc;

        #undef UPDATE_CCOL
      } else { // C_OPTION_MULTIPLY
        fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
        #define UPDATE_CCOL(ccol, acc0, acc1) \
        MM_STOREU_Px(&((ccol)[SIMD_FACTOR*0]), MM_FMADD((acc0), alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px(&((ccol)[SIMD_FACTOR*0]))))); \
        MM_STOREU_Px(&((ccol)[SIMD_FACTOR*1]), MM_FMADD((acc1), alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px(&((ccol)[SIMD_FACTOR*1])))));

        UPDATE_CCOL(C, acc00, acc10);  Prefetch2lines(C); C += ldc;
        UPDATE_CCOL(C, acc01, acc11);  Prefetch2lines(C); C += ldc;
        UPDATE_CCOL(C, acc02, acc12);  Prefetch2lines(C); C += ldc;
        UPDATE_CCOL(C, acc03, acc13);  Prefetch2lines(C); C += ldc;
        UPDATE_CCOL(C, acc04, acc14);  Prefetch2lines(C); C += ldc;

        #undef UPDATE_CCOL
      }
    } else {
      int_vector_t mask = pPrm->mask_b[1];
      if (pPrm->c_option == C_OPTION_UPDATE) {
        #define UPDATE_CCOL(ccol, acc0, acc1) \
        MM_STOREU_Px    (&((ccol)[SIMD_FACTOR*0]),       MM_FMADD((acc0), alpha_ps, MM_LOADU_Px    (&((ccol)[SIMD_FACTOR*0])))); \
        MM_MASKSTOREU_Px(&((ccol)[SIMD_FACTOR*1]), mask, MM_FMADD((acc1), alpha_ps, MM_MASKLOADU_Px(&((ccol)[SIMD_FACTOR*1]), mask)));

        UPDATE_CCOL(C, acc00, acc10);  Prefetch2lines(C); C += ldc;
        UPDATE_CCOL(C, acc01, acc11);  Prefetch2lines(C); C += ldc;
        UPDATE_CCOL(C, acc02, acc12);  Prefetch2lines(C); C += ldc;
        UPDATE_CCOL(C, acc03, acc13);  Prefetch2lines(C); C += ldc;
        UPDATE_CCOL(C, acc04, acc14);  Prefetch2lines(C); C += ldc;

        #undef UPDATE_CCOL
      } else if (pPrm->c_option == C_OPTION_REPLACE) {
        #define UPDATE_CCOL(ccol, acc0, acc1) \
        MM_STOREU_Px    (&((ccol)[SIMD_FACTOR*0]),       MM_MUL_Px((acc0), alpha_ps)); \
        MM_MASKSTOREU_Px(&((ccol)[SIMD_FACTOR*1]), mask, MM_MUL_Px((acc1), alpha_ps));

        UPDATE_CCOL(C, acc00, acc10);  Prefetch2lines(C); C += ldc;
        UPDATE_CCOL(C, acc01, acc11);  Prefetch2lines(C); C += ldc;
        UPDATE_CCOL(C, acc02, acc12);  Prefetch2lines(C); C += ldc;
        UPDATE_CCOL(C, acc03, acc13);  Prefetch2lines(C); C += ldc;
        UPDATE_CCOL(C, acc04, acc14);  Prefetch2lines(C); C += ldc;

        #undef UPDATE_CCOL
      } else { // C_OPTION_MULTIPLY
        fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
        #define UPDATE_CCOL(ccol, acc0, acc1) \
        MM_STOREU_Px    (&((ccol)[SIMD_FACTOR*0]),       MM_FMADD((acc0), alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px    (&((ccol)[SIMD_FACTOR*0]))))); \
        MM_MASKSTOREU_Px(&((ccol)[SIMD_FACTOR*1]), mask, MM_FMADD((acc1), alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(&((ccol)[SIMD_FACTOR*1]), mask))));

        UPDATE_CCOL(C, acc00, acc10);  Prefetch2lines(C); C += ldc;
        UPDATE_CCOL(C, acc01, acc11);  Prefetch2lines(C); C += ldc;
        UPDATE_CCOL(C, acc02, acc12);  Prefetch2lines(C); C += ldc;
        UPDATE_CCOL(C, acc03, acc13);  Prefetch2lines(C); C += ldc;
        UPDATE_CCOL(C, acc04, acc14);  Prefetch2lines(C); C += ldc;

        #undef UPDATE_CCOL
      }
    }
  }

  // handle remaining rows of a - non-interleaved
  for (; m < pPrm->M;  A += kSteps, C += ldc, ++m) {
    const fp_vector_t* Bcol = pPrm->bb;
    const scalar_t*    ARow = (const scalar_t*)(A);
    fp_vector_t acc00 = MM_SETZERO_Px();
    fp_vector_t acc10 = MM_SETZERO_Px();
    fp_vector_t acc01 = MM_SETZERO_Px();
    fp_vector_t acc11 = MM_SETZERO_Px();
    fp_vector_t acc02 = MM_SETZERO_Px();
    fp_vector_t acc12 = MM_SETZERO_Px();
    fp_vector_t acc03 = MM_SETZERO_Px();
    fp_vector_t acc13 = MM_SETZERO_Px();

    _mm_prefetch((char*)(C),             _MM_HINT_T0);
    _mm_prefetch((char*)(C+SIMD_FACTOR), _MM_HINT_T0);

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
    acc00 = MM_ADD_Px(acc00, acc01);
    acc02 = MM_ADD_Px(acc02, acc03);

    acc10 = MM_ADD_Px(acc10, acc11);
    acc12 = MM_ADD_Px(acc12, acc13);

    acc00 = MM_ADD_Px(acc00, acc02);
    acc10 = MM_ADD_Px(acc10, acc12);

    fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
    if (!pPrm->masked_b) {
      if (pPrm->c_option == C_OPTION_UPDATE) {
        MM_STOREU_Px(&C[SIMD_FACTOR*0], MM_FMADD(acc00, alpha_ps, MM_LOADU_Px(&C[SIMD_FACTOR*0])));
        MM_STOREU_Px(&C[SIMD_FACTOR*1], MM_FMADD(acc10, alpha_ps, MM_LOADU_Px(&C[SIMD_FACTOR*1])));
      } else if (pPrm->c_option == C_OPTION_REPLACE) {
        MM_STOREU_Px(&C[SIMD_FACTOR*0], MM_MUL_Px(acc00, alpha_ps));
        MM_STOREU_Px(&C[SIMD_FACTOR*1], MM_MUL_Px(acc10, alpha_ps));
      } else { // C_OPTION_MULTIPLY
        fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
        MM_STOREU_Px(&C[SIMD_FACTOR*0], MM_FMADD(acc00, alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px(&C[SIMD_FACTOR*0]))));
        MM_STOREU_Px(&C[SIMD_FACTOR*1], MM_FMADD(acc10, alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px(&C[SIMD_FACTOR*1]))));
      }
    } else {
      int_vector_t mask = pPrm->mask_b[1];
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
  int kSteps = (unsigned)(nRows - 1) / 4 + 1;
  int m;
  const scalar_t* ARow = (const scalar_t*)(pPrm->aa);
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
    acc00 = MM_ADD_Px(acc00, acc10);
    acc01 = MM_ADD_Px(acc01, acc11);
    acc02 = MM_ADD_Px(acc02, acc12);
    acc03 = MM_ADD_Px(acc03, acc13);
    acc04 = MM_ADD_Px(acc04, acc14);

    fp_vector_t  alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
    int_vector_t mask = pPrm->mask_b[1];

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
    acc00 = MM_ADD_Px(acc00, acc01);
    acc02 = MM_ADD_Px(acc02, acc03);
    acc00 = MM_ADD_Px(acc00, acc02);

    fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
    int_vector_t mask = pPrm->mask_b[1];
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
  int                   nRows)
{
  fp_vector_t* dst = pPrm->bb;
  for (int r = 0; r < nRows; ++r) {
    // 'gcc -O1' does not generate good code for memcpy
    // On the other hand, MSVC does not generate good code for loop
    // Since these couple of lines is performance-critical and can easily cost 6-7% in time
    // I coded it in ugly manner with ifdef, to please each compiler with its preferred construct
    #ifdef _MSC_VER
    memcpy(dst, B, sizeof(*dst)*B_WORDS_PER_ITER);
    #else
    for (int w = 0; w < B_WORDS_PER_ITER; ++w)
      dst[w] = MM_LOADU_Px(&B[w*SIMD_FACTOR]);
    #endif
    B   += ldb;
    dst += B_WORDS_PER_ITER;
  }
}

static void CopyAndTransposeMjWithMask(
  noncblas_sgemm_prm_t* pPrm,
  const scalar_t*       B, int ldb,
  int                   nRows)
{
  fp_vector_t* dst = pPrm->bb;
  int_vector_t mask = pPrm->mask_b[1];
  for (int r = 0; r < nRows; ++r) {
    for (int w = 0; w < B_WORDS_PER_ITER-1; ++w)
      dst[w] = MM_LOADU_Px(&B[w*SIMD_FACTOR]);
    dst[B_WORDS_PER_ITER-1] = MM_MASKLOADU_Px(&B[(B_WORDS_PER_ITER-1)*SIMD_FACTOR], mask);
    B   += ldb;
    dst += B_WORDS_PER_ITER;
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
  int nQuads = (unsigned)(n_cols-1)/4 + 1;
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
static void UpdateC(noncblas_sgemm_prm_t* pPrm, scalar_t *C, int n_cols)
{
  int nRows = pPrm->M;
  int nSimdCols = n_cols / SIMD_FACTOR;
  int ldc = pPrm->ldc;
  const scalar_t* src = pPrm->cc;
  switch (pPrm->c_option) {
    case C_OPTION_UPDATE:
    {
      for (int r = 0; r < nRows; C += ldc, ++r) {
        scalar_t* dst = C;
        for (int c = 0; c < nSimdCols; src += SIMD_FACTOR, dst += SIMD_FACTOR, ++c) {
          MM_STOREU_Px(dst, MM_ADD_Px(MM_LOADU_Px(dst), MM_LOADU_Px(src)));
          _mm_prefetch((char*)&dst[ldc], _MM_HINT_T0); 
        }
      }
    }
    break;

    case C_OPTION_REPLACE:
    {
      for (int r = 0; r < nRows; C += ldc, ++r) {
        scalar_t* dst = C;
        for (int c = 0; c < nSimdCols; src += SIMD_FACTOR, dst += SIMD_FACTOR, ++c) {
          MM_STOREU_Px(dst, MM_LOADU_Px(src));
          _mm_prefetch((char*)&dst[ldc], _MM_HINT_T0); 
        }
      }
    }
    break;

    default: // C_OPTION_MULTIPLY
    {
      fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
      for (int r = 0; r < nRows; C += ldc, ++r) {
        scalar_t* dst = C;
        for (int c = 0; c < nSimdCols; src += SIMD_FACTOR, dst += SIMD_FACTOR, ++c) {
          MM_STOREU_Px(dst, MM_FMADD(MM_LOADU_Px(dst), beta_ps, MM_LOADU_Px(src)));
          _mm_prefetch((char*)&dst[ldc], _MM_HINT_T0); 
        }
      }
    }
    break;
  }
}

static int st_m_step = 0;
static int st_k_step = 0;
#include <stdio.h>

// N>SIMD_FACTOR
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

  int k_step = st_k_step;
  if (k_step == 0) { // auto k_step
    k_step = K_STEP;
    if (M <= SMALL_M_THR) {
      int div = M + N;
      if (SMALL_M_NxK_STEP >= div*28 && SMALL_M_NxK_STEP < div*K_STEP)
        k_step = SMALL_M_NxK_STEP/div;
    }
  }
  int k_Nsteps = (K*4-k_step)/(4*k_step) + 1;
  k_step = k_Nsteps < 2 ? K : ((K-1)/(k_Nsteps*4) + 1) * 4;

  int m_step = st_m_step;
  if (m_step == 0) // auto m_step
    m_step = MxN_BLOCK_SZ/(N*sizeof(scalar_t));

  m_step = m_step > 20 ? m_step : 20;
  int m_Nsteps = m_step > 0 ? (M*2-m_step)/(2*m_step) + 1 : 1;
  m_step = m_Nsteps < 2 ? M : ((M-1)/(m_Nsteps*A_WORDS_PER_ITER) + 1) * A_WORDS_PER_ITER;

  int cc_width = 0;
  if (ldc > 4096/sizeof(scalar_t)/2 && m_step > 50) 
    cc_width = ((4096-1)/(64*A_WORDS_PER_ITER))*64/sizeof(scalar_t);

  const int bb_sz = (SIMD_ELEM_PEC_COL_MJ*k_step)*sizeof(fp_vector_t);
  const int aa_sz = ((m_step*k_step-1)/SIMD_FACTOR + 1)*sizeof(fp_vector_t);
  const int cc_sz = cc_width*m_step*sizeof(scalar_t);
  const int workBufSz = aa_sz + bb_sz + cc_sz;
  const int align = 64;
  // I didn't find a standard portable way to allocate 32-byte aligned buffer
  // So I am doing it in hackish, but reliable way
  char* workBufAlloc = malloc(workBufSz+align);
  uintptr_t workBufAdj = (0-(uintptr_t)(workBufAlloc)) % align;
  char* workBuf = workBufAlloc+workBufAdj;

  noncblas_sgemm_prm_t prm;
  prm.cc = (scalar_t*)(workBuf+0);
  prm.aa = (fp_vector4_t*)(workBuf+cc_sz);
  prm.bb = (fp_vector_t*) (workBuf+cc_sz+aa_sz);
  prm.lda = lda;
  prm.ldc = ldc;
  prm.alpha = alpha;
  prm.beta  = beta;
  memset(&prm.mask_b[0], -1, sizeof(prm.mask_b[0]));
  prm.mask_b[2] = prm.mask_b[0];
  unsigned remW_n = nw*SIMD_FACTOR - N;
  int nwRemMj_masked_b = 0;
  if (remW_n > 0) { // mask off elements of rightmost SIMD word in B and C
    memset((char*)&prm.mask_b[3] - sizeof(*C)*remW_n, 0, sizeof(*C)*remW_n);
    nwRemMj_masked_b = 1;
  }

  memset(&prm.mask_a[0], -1, sizeof(prm.mask_a[0]));
  prm.mask_a[1] = prm.mask_a[0];
  unsigned remW_k = (4 - (unsigned)(K)) % 4;
  if (remW_k > 0) // mask off elements of rightmost fp_vector4_t word in A
    memset((char*)&prm.mask_a[2] - sizeof(*A)*remW_k, 0, sizeof(*A)*remW_k);

  //printf("nMj=%d, nwRemMn=%d, nwRemMj=%d remW_n=%d n_step=%d\n", nMj, nwRemMn, nwRemMj, remW_n, n_step);

  for (int k = 0; k < K; k += k_step) {
    int c_option = C_OPTION_UPDATE;
    if (k == 0 && prm.beta != 1.0f)
      c_option = (prm.beta == 0) ? C_OPTION_REPLACE : C_OPTION_MULTIPLY;
    prm.c_option = c_option;
    int delta_k = k_step < K - k ? k_step : K - k;
    for (int m = 0; m < M; m += prm.M) {
      prm.M = m_step < M - m ? m_step : M - m;

      prm.masked_b = 0;              // all words in use
      prm.mask_b[1] = prm.mask_b[0]; // all words in use
      CopyAndTransposeA(&prm, &A[m*lda + k], delta_k);

      scalar_t *Crow = &C[m*ldc];
      int n;
      if (cc_width == 0) {
        for (n = 0; n < nMj; n += n_step) {
          // process full-width major rectangles
          CopyAndTransposeMj(&prm, &B[k*ldb + n], ldb, delta_k);
          fma256_noncblas_sgemm_core_mj(&prm, &Crow[n], delta_k);
        }
      } else {
        for (n = 0; n < nMj;) {
          int delta_n = cc_width < nMj - n ? cc_width : nMj - n;
          prm.ldc = delta_n;
          prm.c_option = C_OPTION_REPLACE;
          for (int ni = 0; ni < delta_n; ni += n_step) {
            // process full-width major rectangles
            CopyAndTransposeMj(&prm, &B[k*ldb + ni+n], ldb, delta_k);
            fma256_noncblas_sgemm_core_mj(&prm, &prm.cc[ni], delta_k);
          }
          prm.ldc = ldc;
          prm.c_option = c_option;
          UpdateC(&prm, &Crow[n], delta_n);
          n += delta_n;
        }
      }
      if (nwRemMn > 0) {
        if (nwRemMj == 0)
          prm.mask_b[1] = prm.mask_b[2]; // mask for rightmost word
        CopyAndTransposeMnWithMask(&prm, &B[k*ldb + n], ldb, delta_k);
        fma256_noncblas_sgemm_core_mn(&prm, &Crow[n], delta_k);
        n += SIMD_FACTOR;
      }
      if (nwRemMj > 0) {
        prm.mask_b[1] = prm.mask_b[2]; // mask for rightmost word
        prm.masked_b = nwRemMj_masked_b;
        CopyAndTransposeMjWithMask(&prm, &B[k*ldb + n], ldb, delta_k);
        fma256_noncblas_sgemm_core_mj(&prm, &Crow[n], delta_k);
      }
    }
  }
  free(workBufAlloc);
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