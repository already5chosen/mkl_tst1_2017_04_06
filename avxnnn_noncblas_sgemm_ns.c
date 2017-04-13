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
  int           M;
  int           lda;
  int           ldc;
  int           c_option;
  int           masked_b_it;
  scalar_t      alpha;
  scalar_t      beta;
  int_vector_t  mask_b[3]; // [0]= all 1s, [1] = all ones or last SIMD word, [2] = last SIMD word
  fp_vector_t*  bb; // [SIMD_ELEM_PEC_COL_MJ*k_step];
  const scalar_t* A;
} noncblas_sgemm_prm_t;

// major core - inner loop processes 2 SIMD columns of B x 5 rows of A
static void fma256_noncblas_sgemm_core_mj(
 const noncblas_sgemm_prm_t* pPrm,
 scalar_t*                   C,
 int                         n_bIters, // 0 < n_bIters <= N_STEP_MULTIPLIER
 int                         nRows)    // 0 < nRows    <= k_step
{
  int ldc = pPrm->ldc;
  int kFullSteps = (unsigned)(nRows) / 4;
  int kRemSteps  = (unsigned)(nRows) % 4;
  int ldbb   = B_WORDS_PER_ITER*nRows;
  int m;
  const scalar_t* A = pPrm->A;
  int b_itLast = n_bIters - 1;
  int lda1 = pPrm->lda;
  int lda2 = lda1+lda1;
  int lda3 = lda2+lda1;
  int lda4 = lda3+lda1;
  int lda5 = lda4+lda1;
  const ptrdiff_t nextM_preftechDistance = (ldc*A_WORDS_PER_ITER - b_itLast*B_WORDS_PER_ITER*SIMD_FACTOR)*sizeof(*C);
  for (m = 0; m < pPrm->M-A_WORDS_PER_ITER+1;
    A += lda5,
    C += ldc*A_WORDS_PER_ITER,
    m += A_WORDS_PER_ITER) {
    scalar_t* Crow = C;
    for (int b_it = 0; b_it <= b_itLast; Crow += B_WORDS_PER_ITER*SIMD_FACTOR, ++b_it) {
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
      fp_vector_t acc04 = MM_SETZERO_Px();
      fp_vector_t acc14 = MM_SETZERO_Px();

      for (int k = 0; k < kFullSteps; ++k) {
        fp_vector_t a, b0, b1;
        b0 = Bcol[0];
        b1 = Bcol[1];
        Bcol += B_WORDS_PER_ITER;

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
        Bcol += B_WORDS_PER_ITER;

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
        Bcol += B_WORDS_PER_ITER;

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
        Bcol += B_WORDS_PER_ITER;

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
        Bcol += B_WORDS_PER_ITER;

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
          Bcol += B_WORDS_PER_ITER;

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
            Bcol += B_WORDS_PER_ITER;

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

      fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
      scalar_t* CCol = Crow;
      ptrdiff_t preftechDistance = (b_it==b_itLast) ? nextM_preftechDistance : B_WORDS_PER_ITER*SIMD_FACTOR*sizeof(*C);
      #define Prefetch2lines(x) \
        _mm_prefetch((char*)&((x)[SIMD_FACTOR*0])+preftechDistance, _MM_HINT_T0); \
        _mm_prefetch((char*)&((x)[SIMD_FACTOR*1])+preftechDistance, _MM_HINT_T0);

      if (b_it != pPrm->masked_b_it) {
        if (pPrm->c_option == C_OPTION_UPDATE) {
          #define UPDATE_CCOL(ccol, acc0, acc1) \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*0]), MM_FMADD((acc0), alpha_ps, MM_LOADU_Px(&((ccol)[SIMD_FACTOR*0])))); \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*1]), MM_FMADD((acc1), alpha_ps, MM_LOADU_Px(&((ccol)[SIMD_FACTOR*1]))));

          UPDATE_CCOL(CCol, acc00, acc10);  Prefetch2lines(CCol); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11);  Prefetch2lines(CCol); CCol += ldc;
          UPDATE_CCOL(CCol, acc02, acc12);  Prefetch2lines(CCol); CCol += ldc;
          UPDATE_CCOL(CCol, acc03, acc13);  Prefetch2lines(CCol); CCol += ldc;
          UPDATE_CCOL(CCol, acc04, acc14);  Prefetch2lines(CCol);

          #undef UPDATE_CCOL
        } else if (pPrm->c_option == C_OPTION_REPLACE) {
          #define UPDATE_CCOL(ccol, acc0, acc1) \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*0]), MM_MUL_Px((acc0), alpha_ps)); \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*1]), MM_MUL_Px((acc1), alpha_ps));

          UPDATE_CCOL(CCol, acc00, acc10);  Prefetch2lines(CCol); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11);  Prefetch2lines(CCol); CCol += ldc;
          UPDATE_CCOL(CCol, acc02, acc12);  Prefetch2lines(CCol); CCol += ldc;
          UPDATE_CCOL(CCol, acc03, acc13);  Prefetch2lines(CCol); CCol += ldc;
          UPDATE_CCOL(CCol, acc04, acc14);  Prefetch2lines(CCol);

          #undef UPDATE_CCOL
        } else { // C_OPTION_MULTIPLY
          fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
          #define UPDATE_CCOL(ccol, acc0, acc1) \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*0]), MM_FMADD((acc0), alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px(&((ccol)[SIMD_FACTOR*0]))))); \
          MM_STOREU_Px(&((ccol)[SIMD_FACTOR*1]), MM_FMADD((acc1), alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px(&((ccol)[SIMD_FACTOR*1])))));

          UPDATE_CCOL(CCol, acc00, acc10);  Prefetch2lines(CCol); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11);  Prefetch2lines(CCol); CCol += ldc;
          UPDATE_CCOL(CCol, acc02, acc12);  Prefetch2lines(CCol); CCol += ldc;
          UPDATE_CCOL(CCol, acc03, acc13);  Prefetch2lines(CCol); CCol += ldc;
          UPDATE_CCOL(CCol, acc04, acc14);  Prefetch2lines(CCol);

          #undef UPDATE_CCOL
        }
      } else {
        int_vector_t mask = pPrm->mask_b[1];
        if (pPrm->c_option == C_OPTION_UPDATE) {
          #define UPDATE_CCOL(ccol, acc0, acc1) \
          MM_STOREU_Px    (&((ccol)[SIMD_FACTOR*0]),       MM_FMADD((acc0), alpha_ps, MM_LOADU_Px    (&((ccol)[SIMD_FACTOR*0])))); \
          MM_MASKSTOREU_Px(&((ccol)[SIMD_FACTOR*1]), mask, MM_FMADD((acc1), alpha_ps, MM_MASKLOADU_Px(&((ccol)[SIMD_FACTOR*1]), mask)));

          UPDATE_CCOL(CCol, acc00, acc10);  Prefetch2lines(CCol); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11);  Prefetch2lines(CCol); CCol += ldc;
          UPDATE_CCOL(CCol, acc02, acc12);  Prefetch2lines(CCol); CCol += ldc;
          UPDATE_CCOL(CCol, acc03, acc13);  Prefetch2lines(CCol); CCol += ldc;
          UPDATE_CCOL(CCol, acc04, acc14);  Prefetch2lines(CCol);

          #undef UPDATE_CCOL
        } else if (pPrm->c_option == C_OPTION_REPLACE) {
          #define UPDATE_CCOL(ccol, acc0, acc1) \
          MM_STOREU_Px    (&((ccol)[SIMD_FACTOR*0]),       MM_MUL_Px((acc0), alpha_ps)); \
          MM_MASKSTOREU_Px(&((ccol)[SIMD_FACTOR*1]), mask, MM_MUL_Px((acc1), alpha_ps));

          UPDATE_CCOL(CCol, acc00, acc10);  Prefetch2lines(CCol); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11);  Prefetch2lines(CCol); CCol += ldc;
          UPDATE_CCOL(CCol, acc02, acc12);  Prefetch2lines(CCol); CCol += ldc;
          UPDATE_CCOL(CCol, acc03, acc13);  Prefetch2lines(CCol); CCol += ldc;
          UPDATE_CCOL(CCol, acc04, acc14);  Prefetch2lines(CCol);

          #undef UPDATE_CCOL
        } else { // C_OPTION_MULTIPLY
          fp_vector_t beta_ps = MM_BROADCAST_Sx(&pPrm->beta);
          #define UPDATE_CCOL(ccol, acc0, acc1) \
          MM_STOREU_Px    (&((ccol)[SIMD_FACTOR*0]),       MM_FMADD((acc0), alpha_ps, MM_MUL_Px(beta_ps, MM_LOADU_Px    (&((ccol)[SIMD_FACTOR*0]))))); \
          MM_MASKSTOREU_Px(&((ccol)[SIMD_FACTOR*1]), mask, MM_FMADD((acc1), alpha_ps, MM_MUL_Px(beta_ps, MM_MASKLOADU_Px(&((ccol)[SIMD_FACTOR*1]), mask))));

          UPDATE_CCOL(CCol, acc00, acc10);  Prefetch2lines(CCol); CCol += ldc;
          UPDATE_CCOL(CCol, acc01, acc11);  Prefetch2lines(CCol); CCol += ldc;
          UPDATE_CCOL(CCol, acc02, acc12);  Prefetch2lines(CCol); CCol += ldc;
          UPDATE_CCOL(CCol, acc03, acc13);  Prefetch2lines(CCol); CCol += ldc;
          UPDATE_CCOL(CCol, acc04, acc14);  Prefetch2lines(CCol);

          #undef UPDATE_CCOL
        }
      }
    }
  }

  // handle remaining rows of A
  for (; m < pPrm->M;  A += lda1, C += ldc, ++m) {
    scalar_t* Crow = C;
    for (int b_it = 0; b_it <= b_itLast; Crow += B_WORDS_PER_ITER*SIMD_FACTOR, ++b_it) {
      const fp_vector_t* Bcol = &pPrm->bb[ldbb*b_it];
      const scalar_t*    ARow = A;
      fp_vector_t acc00 = MM_SETZERO_Px();
      fp_vector_t acc10 = MM_SETZERO_Px();
      fp_vector_t acc01 = MM_SETZERO_Px();
      fp_vector_t acc11 = MM_SETZERO_Px();
      fp_vector_t acc02 = MM_SETZERO_Px();
      fp_vector_t acc12 = MM_SETZERO_Px();
      fp_vector_t acc03 = MM_SETZERO_Px();
      fp_vector_t acc13 = MM_SETZERO_Px();

      _mm_prefetch((char*)(Crow),             _MM_HINT_T0);
      _mm_prefetch((char*)(Crow+SIMD_FACTOR), _MM_HINT_T0);

      for (int k = 0; k < kFullSteps; ++k) {
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

      if (kRemSteps != 0) {
        fp_vector_t a;

        a = MM_BROADCAST_Sx(&ARow[0]);
        acc00 = MM_FMADD(a, Bcol[0], acc00);
        acc10 = MM_FMADD(a, Bcol[1], acc10);
        if (kRemSteps != 1) {
          Bcol += B_WORDS_PER_ITER;

          a = MM_BROADCAST_Sx(&ARow[1]);
          acc01 = MM_FMADD(a, Bcol[0], acc01);
          acc11 = MM_FMADD(a, Bcol[1], acc11);
          if (kRemSteps != 2) {
            Bcol += B_WORDS_PER_ITER;

            a = MM_BROADCAST_Sx(&ARow[2]);
            acc02 = MM_FMADD(a, Bcol[0], acc02);
            acc12 = MM_FMADD(a, Bcol[1], acc12);
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
        int_vector_t mask = pPrm->mask_b[1];
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
}

// minor core - inner loop processes 1 SIMD columns of B x 5 rows of A
static void fma256_noncblas_sgemm_core_mn(
 const noncblas_sgemm_prm_t* pPrm,
 scalar_t*                   C,
 int                         nRows)    // 0 < nRows <= k_step
{
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
  for (m = 0; m < pPrm->M - A_WORDS_PER_ITER + 1; A += lda5, m += A_WORDS_PER_ITER) {
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
    const scalar_t* ARow = A;
    for (int k = 0; k < kFullSteps; ++k) {
      fp_vector_t b;

      b = Bcol[0];
      acc00 = MM_FMADD(MM_BROADCAST_Sx(&ARow[0]), b, acc00);
      acc01 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda1 + 0]), b, acc01);
      acc02 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda2 + 0]), b, acc02);
      acc03 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda3 + 0]), b, acc03);
      acc04 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda4 + 0]), b, acc04);

      b = Bcol[1];
      acc10 = MM_FMADD(MM_BROADCAST_Sx(&ARow[1]), b, acc10);
      acc11 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda1 + 1]), b, acc11);
      acc12 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda2 + 1]), b, acc12);
      acc13 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda3 + 1]), b, acc13);
      acc14 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda4 + 1]), b, acc14);

      b = Bcol[2];
      acc00 = MM_FMADD(MM_BROADCAST_Sx(&ARow[2]), b, acc00);
      acc01 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda1 + 2]), b, acc01);
      acc02 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda2 + 2]), b, acc02);
      acc03 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda3 + 2]), b, acc03);
      acc04 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda4 + 2]), b, acc04);

      b = Bcol[3];
      acc10 = MM_FMADD(MM_BROADCAST_Sx(&ARow[3]), b, acc10);
      acc11 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda1 + 3]), b, acc11);
      acc12 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda2 + 3]), b, acc12);
      acc13 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda3 + 3]), b, acc13);
      acc14 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda4 + 3]), b, acc14);

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
      acc04 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda4 + 0]), b, acc04);

      if (kRemSteps != 1) {
        b = Bcol[1];
        acc10 = MM_FMADD(MM_BROADCAST_Sx(&ARow[1]), b, acc10);
        acc11 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda1 + 1]), b, acc11);
        acc12 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda2 + 1]), b, acc12);
        acc13 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda3 + 1]), b, acc13);
        acc14 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda4 + 1]), b, acc14);

        if (kRemSteps != 2) {
          b = Bcol[2];
          acc00 = MM_FMADD(MM_BROADCAST_Sx(&ARow[2]), b, acc00);
          acc01 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda1 + 2]), b, acc01);
          acc02 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda2 + 2]), b, acc02);
          acc03 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda3 + 2]), b, acc03);
          acc04 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda4 + 2]), b, acc04);
        }
      }
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

  int k_step = K;
  int m_step = M;

  const int bb_sz = SIMD_ELEM_PEC_COL_MJ*K;
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
  memset(&prm.mask_b[0], -1, sizeof(prm.mask_b[0]));
  prm.mask_b[2] = prm.mask_b[0];
  unsigned remW_n = nw*SIMD_FACTOR - N;
  int nwRemMj_masked_b_it = -1;
  if (remW_n > 0) { // mask off elements of rightmost SIMD word in B and C
    memset((char*)&prm.mask_b[3] - sizeof(*C)*remW_n, 0, sizeof(*C)*remW_n);
    nwRemMj_masked_b_it = nwRemMj - 1;
  }

  //printf("nMj=%d, nwRemMn=%d, nwRemMj=%d remW_n=%d n_step=%d\n", nMj, nwRemMn, nwRemMj, remW_n, n_step);

  for (int k = 0; k < K; k += k_step) {
    prm.c_option = C_OPTION_UPDATE;
    if (k==0 && prm.beta != 1.0f)
      prm.c_option = (prm.beta == 0) ? C_OPTION_REPLACE : C_OPTION_MULTIPLY;
    int delta_k = k_step < K - k ? k_step : K - k;
    for (int m = 0; m < M; m += prm.M) {
      prm.M = m_step < M - m ? m_step : M - m;

      prm.masked_b_it = -1;          // all words in use
      prm.mask_b[1] = prm.mask_b[0]; // all words in use
      prm.A = &A[m*lda+k];

      scalar_t *Crow = &C[m*ldc];
      int n;
      for (n = 0; n < nMj; n += n_step) {
        // process full-width major rectangles
        CopyAndTransposeMj(&prm, &B[k*ldb + n], ldb, N_STEP_MULTIPLIER, delta_k);
        fma256_noncblas_sgemm_core_mj(&prm, &Crow[n], N_STEP_MULTIPLIER, delta_k);
      }
      if (nwRemMn > 0) {
        if (nwRemMj == 0)
          prm.mask_b[1] = prm.mask_b[2]; // mask for rightmost word
        CopyAndTransposeMnWithMask(&prm, &B[k*ldb + n], ldb, delta_k);
        fma256_noncblas_sgemm_core_mn(&prm, &Crow[n], delta_k);
        n += SIMD_FACTOR;
      }
      if (nwRemMj > 0) {
        prm.mask_b[1]   = prm.mask_b[2]; // mask for rightmost word
        prm.masked_b_it = nwRemMj_masked_b_it;
        CopyAndTransposeMjWithMask(&prm, &B[k*ldb + n], ldb, nwRemMj, delta_k);
        fma256_noncblas_sgemm_core_mj(&prm, &Crow[n], nwRemMj, delta_k);
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