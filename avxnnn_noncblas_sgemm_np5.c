// both A and B unbuffered in the inner loop
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
  int             M;
  int             lda;
  int             ldb;
  int             ldc;
  int             c_option;
  int             masked_b_it;
  scalar_t        alpha;
  scalar_t        beta;
  int_vector_t    mask_n;
  const scalar_t* A;
  const scalar_t* B;
} noncblas_sgemm_prm_t;

// major core - inner loop processes 2 SIMD columns of B x 5 rows of A
static void fma256_noncblas_sgemm_core_mj(
 const noncblas_sgemm_prm_t* pPrm,
 scalar_t*                   C,
 int                         n_bIters, // 0 < n_bIters <= N_STEP_MULTIPLIER
 int                         nRows)    // 7 < nRows    <= k_step
{
  int ldc = pPrm->ldc;
  int kFullSteps = (unsigned)(nRows-1) / 4;
  int kRemSteps  = (unsigned)(nRows-1) % 4;
  int m;
  const scalar_t* A = pPrm->A;
  int b_itLast = n_bIters - 1;
  int lda1 = pPrm->lda;
  int lda2 = lda1+lda1;
  int lda3 = lda2+lda1;
  int lda4 = lda3+lda1;
  int lda5 = lda4+lda1;
  int ldb1 = pPrm->ldb;
  int ldb2 = ldb1+ldb1;
  const ptrdiff_t nextM_preftechDistance = (ldc*A_WORDS_PER_ITER - b_itLast*B_WORDS_PER_ITER*SIMD_FACTOR)*sizeof(*C);
  for (m = 0; m < pPrm->M-A_WORDS_PER_ITER+1;
    A += lda5,
    C += ldc*A_WORDS_PER_ITER,
    m += A_WORDS_PER_ITER) {
    for (int b_it = 0, bc_offset = 0; b_it <= b_itLast; bc_offset += B_WORDS_PER_ITER*SIMD_FACTOR, ++b_it) {
      const scalar_t* ARow = A;
      const scalar_t* Bcol = &pPrm->B[bc_offset];
      fp_vector_t a = MM_BROADCAST_Sx(&ARow[0]);

      if (b_it != pPrm->masked_b_it) {
        fp_vector_t b0 = MM_LOADU_Px(&Bcol[0*SIMD_FACTOR]);
        fp_vector_t b1 = MM_LOADU_Px(&Bcol[1*SIMD_FACTOR]);
        Bcol += ldb1;

        a = MM_BROADCAST_Sx(&ARow[0]);
        fp_vector_t acc00 = MM_MUL_Px(a, b0);
        fp_vector_t acc10 = MM_MUL_Px(a, b1);

        a = MM_BROADCAST_Sx(&ARow[lda1+0]);
        fp_vector_t acc01 = MM_MUL_Px(a, b0);
        fp_vector_t acc11 = MM_MUL_Px(a, b1);

        a = MM_BROADCAST_Sx(&ARow[lda2+0]);
        fp_vector_t acc02 = MM_MUL_Px(a, b0);
        fp_vector_t acc12 = MM_MUL_Px(a, b1);

        a = MM_BROADCAST_Sx(&ARow[lda3+0]);
        fp_vector_t acc03 = MM_MUL_Px(a, b0);
        fp_vector_t acc13 = MM_MUL_Px(a, b1);

        a = MM_BROADCAST_Sx(&ARow[lda4+0]);
        fp_vector_t acc04 = MM_MUL_Px(a, b0);
        fp_vector_t acc14 = MM_MUL_Px(a, b1);
        ARow += 1;

        int k = kFullSteps;
        do {
          b0 = MM_LOADU_Px(&Bcol[0*SIMD_FACTOR]);
          b1 = MM_LOADU_Px(&Bcol[1*SIMD_FACTOR]);
          _mm_prefetch((char*)&Bcol[2*SIMD_FACTOR], _MM_HINT_T0);
          _mm_prefetch((char*)&Bcol[3*SIMD_FACTOR], _MM_HINT_T0);

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

          b0 = MM_LOADU_Px(&Bcol[ldb1+0*SIMD_FACTOR]);
          b1 = MM_LOADU_Px(&Bcol[ldb1+1*SIMD_FACTOR]);
          _mm_prefetch((char*)&Bcol[ldb1+2*SIMD_FACTOR], _MM_HINT_T0);
          _mm_prefetch((char*)&Bcol[ldb1+3*SIMD_FACTOR], _MM_HINT_T0);
          Bcol += ldb2;

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

          b0 = MM_LOADU_Px(&Bcol[0*SIMD_FACTOR]);
          b1 = MM_LOADU_Px(&Bcol[1*SIMD_FACTOR]);
          _mm_prefetch((char*)&Bcol[2*SIMD_FACTOR], _MM_HINT_T0);
          _mm_prefetch((char*)&Bcol[3*SIMD_FACTOR], _MM_HINT_T0);

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

          b0 = MM_LOADU_Px(&Bcol[ldb1+0*SIMD_FACTOR]);
          b1 = MM_LOADU_Px(&Bcol[ldb1+1*SIMD_FACTOR]);
          _mm_prefetch((char*)&Bcol[ldb1+2*SIMD_FACTOR], _MM_HINT_T0);
          _mm_prefetch((char*)&Bcol[ldb1+3*SIMD_FACTOR], _MM_HINT_T0);
          Bcol += ldb2;

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
        } while (--k);

        if (kRemSteps != 0) {
          b0 = MM_LOADU_Px(&Bcol[0*SIMD_FACTOR]);
          b1 = MM_LOADU_Px(&Bcol[1*SIMD_FACTOR]);

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
            Bcol += ldb1;
            b0 = MM_LOADU_Px(&Bcol[0*SIMD_FACTOR]);
            b1 = MM_LOADU_Px(&Bcol[1*SIMD_FACTOR]);

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
              Bcol += ldb1;
              b0 = MM_LOADU_Px(&Bcol[0*SIMD_FACTOR]);
              b1 = MM_LOADU_Px(&Bcol[1*SIMD_FACTOR]);

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
        scalar_t* CCol = &C[bc_offset];
        ptrdiff_t preftechDistance = (b_it==b_itLast) ? nextM_preftechDistance : B_WORDS_PER_ITER*SIMD_FACTOR*sizeof(*C);
        #define Prefetch2lines(x) \
          _mm_prefetch((char*)&((x)[SIMD_FACTOR*0])+preftechDistance, _MM_HINT_T0); \
          _mm_prefetch((char*)&((x)[SIMD_FACTOR*1])+preftechDistance, _MM_HINT_T0);

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
        int_vector_t mask = pPrm->mask_n;
        fp_vector_t b0 = MM_LOADU_Px(&Bcol[0*SIMD_FACTOR]);
        fp_vector_t b1 = MM_MASKLOADU_Px(&Bcol[1*SIMD_FACTOR], mask);
        Bcol += ldb1;

        fp_vector_t acc00 = MM_MUL_Px(a, b0);
        fp_vector_t acc10 = MM_MUL_Px(a, b1);

        a = MM_BROADCAST_Sx(&ARow[lda1+0]);
        fp_vector_t acc01 = MM_MUL_Px(a, b0);
        fp_vector_t acc11 = MM_MUL_Px(a, b1);

        a = MM_BROADCAST_Sx(&ARow[lda2+0]);
        fp_vector_t acc02 = MM_MUL_Px(a, b0);
        fp_vector_t acc12 = MM_MUL_Px(a, b1);

        a = MM_BROADCAST_Sx(&ARow[lda3+0]);
        fp_vector_t acc03 = MM_MUL_Px(a, b0);
        fp_vector_t acc13 = MM_MUL_Px(a, b1);

        a = MM_BROADCAST_Sx(&ARow[lda4+0]);
        fp_vector_t acc04 = MM_MUL_Px(a, b0);
        fp_vector_t acc14 = MM_MUL_Px(a, b1);
        ARow += 1;

        int k = kFullSteps;
        do {
          b0 = MM_LOADU_Px(&Bcol[0*SIMD_FACTOR]);
          b1 = MM_MASKLOADU_Px(&Bcol[1*SIMD_FACTOR], mask);

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

          b0 = MM_LOADU_Px(&Bcol[ldb1+0*SIMD_FACTOR]);
          b1 = MM_MASKLOADU_Px(&Bcol[ldb1+1*SIMD_FACTOR], mask);
          Bcol += ldb2;

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

          b0 = MM_LOADU_Px(&Bcol[0*SIMD_FACTOR]);
          b1 = MM_MASKLOADU_Px(&Bcol[1*SIMD_FACTOR], mask);

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

          b0 = MM_LOADU_Px(&Bcol[ldb1+0*SIMD_FACTOR]);
          b1 = MM_MASKLOADU_Px(&Bcol[ldb1+1*SIMD_FACTOR], mask);
          Bcol += ldb2;

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
        } while (--k);

        if (kRemSteps != 0) {
          b0 = MM_LOADU_Px(&Bcol[0*SIMD_FACTOR]);
          b1 = MM_MASKLOADU_Px(&Bcol[1*SIMD_FACTOR], mask);

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
            Bcol += ldb1;
            b0 = MM_LOADU_Px(&Bcol[0*SIMD_FACTOR]);
            b1 = MM_MASKLOADU_Px(&Bcol[1*SIMD_FACTOR], mask);

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
              Bcol += ldb1;
              b0 = MM_LOADU_Px(&Bcol[0*SIMD_FACTOR]);
              b1 = MM_MASKLOADU_Px(&Bcol[1*SIMD_FACTOR], mask);

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
        scalar_t* CCol = &C[bc_offset];
        ptrdiff_t preftechDistance = (b_it==b_itLast) ? nextM_preftechDistance : B_WORDS_PER_ITER*SIMD_FACTOR*sizeof(*C);

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
    for (int b_it = 0, bc_offset = 0; b_it <= b_itLast; bc_offset += B_WORDS_PER_ITER*SIMD_FACTOR, ++b_it) {
      fp_vector_t acc01 = MM_SETZERO_Px();
      fp_vector_t acc11 = MM_SETZERO_Px();
      fp_vector_t acc02 = MM_SETZERO_Px();
      fp_vector_t acc12 = MM_SETZERO_Px();
      fp_vector_t acc03 = MM_SETZERO_Px();
      fp_vector_t acc13 = MM_SETZERO_Px();

      scalar_t* Crow = &C[bc_offset];
      _mm_prefetch((char*)(Crow),             _MM_HINT_T0);
      _mm_prefetch((char*)(Crow+SIMD_FACTOR), _MM_HINT_T0);

      const scalar_t* ARow = A;
      const scalar_t* Bcol = &pPrm->B[bc_offset];
      fp_vector_t a = MM_BROADCAST_Sx(&ARow[0]);
      ARow += 1;
      fp_vector_t acc00 = MM_MUL_Px(a, MM_LOADU_Px(&Bcol[0*SIMD_FACTOR]));
      if (b_it != pPrm->masked_b_it) {
        fp_vector_t acc10 = MM_MUL_Px(a, MM_LOADU_Px(&Bcol[1*SIMD_FACTOR]));
        Bcol += ldb1;

        int k = kFullSteps;
        do {
          a = MM_BROADCAST_Sx(&ARow[0]);
          acc00 = MM_FMADD(a, MM_LOADU_Px(&Bcol[0*SIMD_FACTOR]), acc00);
          acc10 = MM_FMADD(a, MM_LOADU_Px(&Bcol[1*SIMD_FACTOR]), acc10);
          Bcol += ldb1;

          a = MM_BROADCAST_Sx(&ARow[1]);
          acc01 = MM_FMADD(a, MM_LOADU_Px(&Bcol[0*SIMD_FACTOR]), acc01);
          acc11 = MM_FMADD(a, MM_LOADU_Px(&Bcol[1*SIMD_FACTOR]), acc11);
          Bcol += ldb1;

          a = MM_BROADCAST_Sx(&ARow[2]);
          acc02 = MM_FMADD(a, MM_LOADU_Px(&Bcol[0*SIMD_FACTOR]), acc02);
          acc12 = MM_FMADD(a, MM_LOADU_Px(&Bcol[1*SIMD_FACTOR]), acc12);
          Bcol += ldb1;

          a = MM_BROADCAST_Sx(&ARow[3]);
          acc03 = MM_FMADD(a, MM_LOADU_Px(&Bcol[0*SIMD_FACTOR]), acc03);
          acc13 = MM_FMADD(a, MM_LOADU_Px(&Bcol[1*SIMD_FACTOR]), acc13);
          Bcol += ldb1;

          ARow += 4;
        } while (--k);

        if (kRemSteps != 0) {
          a = MM_BROADCAST_Sx(&ARow[0]);
          acc00 = MM_FMADD(a, MM_LOADU_Px(&Bcol[0*SIMD_FACTOR]), acc00);
          acc10 = MM_FMADD(a, MM_LOADU_Px(&Bcol[1*SIMD_FACTOR]), acc10);
          if (kRemSteps != 1) {
            Bcol += ldb1;

            a = MM_BROADCAST_Sx(&ARow[1]);
            acc01 = MM_FMADD(a, MM_LOADU_Px(&Bcol[0*SIMD_FACTOR]), acc01);
            acc11 = MM_FMADD(a, MM_LOADU_Px(&Bcol[1*SIMD_FACTOR]), acc11);
            if (kRemSteps != 2) {
              Bcol += ldb1;

              a = MM_BROADCAST_Sx(&ARow[2]);
              acc02 = MM_FMADD(a, MM_LOADU_Px(&Bcol[0*SIMD_FACTOR]), acc02);
              acc12 = MM_FMADD(a, MM_LOADU_Px(&Bcol[1*SIMD_FACTOR]), acc12);
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
        fp_vector_t acc10 = MM_MUL_Px(a, MM_MASKLOADU_Px(&Bcol[1*SIMD_FACTOR], mask));
        Bcol += ldb1;

        int k = kFullSteps;
        do {
          a = MM_BROADCAST_Sx(&ARow[0]);
          acc00 = MM_FMADD(a, MM_LOADU_Px(&Bcol[0*SIMD_FACTOR]), acc00);
          acc10 = MM_FMADD(a, MM_MASKLOADU_Px(&Bcol[1*SIMD_FACTOR], mask), acc10);
          Bcol += ldb1;

          a = MM_BROADCAST_Sx(&ARow[1]);
          acc01 = MM_FMADD(a, MM_LOADU_Px(&Bcol[0*SIMD_FACTOR]), acc01);
          acc11 = MM_FMADD(a, MM_MASKLOADU_Px(&Bcol[1*SIMD_FACTOR], mask), acc11);
          Bcol += ldb1;

          a = MM_BROADCAST_Sx(&ARow[2]);
          acc02 = MM_FMADD(a, MM_LOADU_Px(&Bcol[0*SIMD_FACTOR]), acc02);
          acc12 = MM_FMADD(a, MM_MASKLOADU_Px(&Bcol[1*SIMD_FACTOR], mask), acc12);
          Bcol += ldb1;

          a = MM_BROADCAST_Sx(&ARow[3]);
          acc03 = MM_FMADD(a, MM_LOADU_Px(&Bcol[0*SIMD_FACTOR]), acc03);
          acc13 = MM_FMADD(a, MM_MASKLOADU_Px(&Bcol[1*SIMD_FACTOR], mask), acc13);
          Bcol += ldb1;

          ARow += 4;
        } while (--k);

        if (kRemSteps != 0) {
          a = MM_BROADCAST_Sx(&ARow[0]);
          acc00 = MM_FMADD(a, MM_LOADU_Px(&Bcol[0*SIMD_FACTOR]), acc00);
          acc10 = MM_FMADD(a, MM_MASKLOADU_Px(&Bcol[1*SIMD_FACTOR], mask), acc10);
          if (kRemSteps != 1) {
            Bcol += ldb1;

            a = MM_BROADCAST_Sx(&ARow[1]);
            acc01 = MM_FMADD(a, MM_LOADU_Px(&Bcol[0*SIMD_FACTOR]), acc01);
            acc11 = MM_FMADD(a, MM_MASKLOADU_Px(&Bcol[1*SIMD_FACTOR], mask), acc11);
            if (kRemSteps != 2) {
              Bcol += ldb1;

              a = MM_BROADCAST_Sx(&ARow[2]);
              acc02 = MM_FMADD(a, MM_LOADU_Px(&Bcol[0*SIMD_FACTOR]), acc02);
              acc12 = MM_FMADD(a, MM_MASKLOADU_Px(&Bcol[1*SIMD_FACTOR], mask), acc12);
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
  int ldb1 = pPrm->ldb;
  const scalar_t* A = pPrm->A;
  int_vector_t mask = pPrm->mask_n;
  for (m = 0; m < pPrm->M - A_WORDS_PER_ITER + 1; A += lda5, m += A_WORDS_PER_ITER) {
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
    const scalar_t* Bcol = pPrm->B;
    for (int k = 0; k < kFullSteps; ++k) {
      fp_vector_t b;

      b = MM_MASKLOADU_Px(Bcol, mask); Bcol += ldb1;
      acc00 = MM_FMADD(MM_BROADCAST_Sx(&ARow[0]), b, acc00);
      acc01 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda1 + 0]), b, acc01);
      acc02 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda2 + 0]), b, acc02);
      acc03 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda3 + 0]), b, acc03);
      acc04 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda4 + 0]), b, acc04);

      b = MM_MASKLOADU_Px(Bcol, mask); Bcol += ldb1;
      acc10 = MM_FMADD(MM_BROADCAST_Sx(&ARow[1]), b, acc10);
      acc11 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda1 + 1]), b, acc11);
      acc12 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda2 + 1]), b, acc12);
      acc13 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda3 + 1]), b, acc13);
      acc14 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda4 + 1]), b, acc14);

      b = MM_MASKLOADU_Px(Bcol, mask); Bcol += ldb1;
      acc00 = MM_FMADD(MM_BROADCAST_Sx(&ARow[2]), b, acc00);
      acc01 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda1 + 2]), b, acc01);
      acc02 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda2 + 2]), b, acc02);
      acc03 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda3 + 2]), b, acc03);
      acc04 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda4 + 2]), b, acc04);

      b = MM_MASKLOADU_Px(Bcol, mask); Bcol += ldb1;
      acc10 = MM_FMADD(MM_BROADCAST_Sx(&ARow[3]), b, acc10);
      acc11 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda1 + 3]), b, acc11);
      acc12 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda2 + 3]), b, acc12);
      acc13 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda3 + 3]), b, acc13);
      acc14 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda4 + 3]), b, acc14);

      ARow += 4;
    }
    if (kRemSteps != 0) {
      fp_vector_t b;

      b = MM_MASKLOADU_Px(Bcol, mask);
      acc00 = MM_FMADD(MM_BROADCAST_Sx(&ARow[0]), b, acc00);
      acc01 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda1 + 0]), b, acc01);
      acc02 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda2 + 0]), b, acc02);
      acc03 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda3 + 0]), b, acc03);
      acc04 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda4 + 0]), b, acc04);

      if (kRemSteps != 1) {
        Bcol += ldb1;
        b = MM_MASKLOADU_Px(Bcol, mask);
        acc10 = MM_FMADD(MM_BROADCAST_Sx(&ARow[1]), b, acc10);
        acc11 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda1 + 1]), b, acc11);
        acc12 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda2 + 1]), b, acc12);
        acc13 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda3 + 1]), b, acc13);
        acc14 = MM_FMADD(MM_BROADCAST_Sx(&ARow[lda4 + 1]), b, acc14);

        if (kRemSteps != 2) {
          Bcol += ldb1;
          b = MM_MASKLOADU_Px(Bcol, mask);
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

  // handle remaining rows of a
  for (; m < pPrm->M; A += lda1, ++m) {
    fp_vector_t acc00 = MM_SETZERO_Px();
    fp_vector_t acc01 = MM_SETZERO_Px();
    fp_vector_t acc02 = MM_SETZERO_Px();
    fp_vector_t acc03 = MM_SETZERO_Px();

    _mm_prefetch((char*)(C), _MM_HINT_T0);
    const scalar_t* ARow = A;
    const scalar_t* Bcol = pPrm->B;
    for (int k = 0; k < kFullSteps; ++k) {
      acc00 = MM_FMADD(MM_BROADCAST_Sx(&ARow[0]), MM_MASKLOADU_Px(Bcol, mask), acc00); Bcol += ldb1;
      acc01 = MM_FMADD(MM_BROADCAST_Sx(&ARow[1]), MM_MASKLOADU_Px(Bcol, mask), acc01); Bcol += ldb1;
      acc02 = MM_FMADD(MM_BROADCAST_Sx(&ARow[2]), MM_MASKLOADU_Px(Bcol, mask), acc02); Bcol += ldb1;
      acc03 = MM_FMADD(MM_BROADCAST_Sx(&ARow[3]), MM_MASKLOADU_Px(Bcol, mask), acc03); Bcol += ldb1;
      ARow += 4;
    }
    if (kRemSteps != 0) {
      acc00 = MM_FMADD(MM_BROADCAST_Sx(&ARow[0]), MM_MASKLOADU_Px(Bcol, mask), acc00);
      if (kRemSteps != 1) {
        Bcol += ldb1;
        acc01 = MM_FMADD(MM_BROADCAST_Sx(&ARow[1]), MM_MASKLOADU_Px(Bcol, mask), acc01);
        if (kRemSteps != 2) {
          Bcol += ldb1;
          acc02 = MM_FMADD(MM_BROADCAST_Sx(&ARow[2]), MM_MASKLOADU_Px(Bcol, mask), acc02);
        }
      }
    }
    acc00 = MM_ADD_Px(acc00, acc01);
    acc02 = MM_ADD_Px(acc02, acc03);
    acc00 = MM_ADD_Px(acc00, acc02);

    fp_vector_t alpha_ps = MM_BROADCAST_Sx(&pPrm->alpha);
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
  int nwMj    = ((nw - 1) / SIMD_ELEM_PEC_COL_MJ) * SIMD_ELEM_PEC_COL_MJ;
  int nwRem   = nw - nwMj;
  int nwRemMj = nwRem / B_WORDS_PER_ITER;
  int nwRemMn = nwRem - nwRemMj*B_WORDS_PER_ITER;
  int nMj     = nwMj * SIMD_FACTOR;

  const int K_STEP_NOM = 200;
  const int K_STEP_MAX = (K_STEP_NOM/8)*12;
  int k_step = K > K_STEP_MAX ? K_STEP_NOM : K;
  int m_step = M;

  noncblas_sgemm_prm_t prm;
  prm.lda = lda;
  prm.ldb = ldb;
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

  for (int k = 0; k < K; k += k_step) {
    prm.c_option = C_OPTION_UPDATE;
    if (k==0 && prm.beta != 1.0f)
      prm.c_option = (prm.beta == 0) ? C_OPTION_REPLACE : C_OPTION_MULTIPLY;
    int delta_k = K - k;
    if (delta_k > k_step) {
      if (delta_k < K_STEP_MAX)
        k_step = ((unsigned)(delta_k-1)/8 + 1)*4;
      delta_k = k_step;
    }
    for (int m = 0; m < M; m += prm.M) {
      prm.M = m_step < M - m ? m_step : M - m;

      prm.masked_b_it = -1;  // all words in use
      prm.A = &A[m*lda+k];

      scalar_t *Crow = &C[m*ldc];
      const scalar_t *Brow = &B[k*ldb];
      int n;
      for (n = 0; n < nMj; n += n_step) {
        // process full-width major rectangles
        prm.B = &Brow[n];
        fma256_noncblas_sgemm_core_mj(&prm, &Crow[n], N_STEP_MULTIPLIER, delta_k);
      }
      if (nwRemMj > 0) {
        prm.masked_b_it = nwRemMj_masked_b_it;
        prm.B = &Brow[n];
        fma256_noncblas_sgemm_core_mj(&prm, &Crow[n], nwRemMj, delta_k);
        n += SIMD_FACTOR*B_WORDS_PER_ITER*nwRemMj;
      }
      if (nwRemMn != 0) {
        prm.B = &Brow[n];
        fma256_noncblas_sgemm_core_mn(&prm, &Crow[n], delta_k);
      }
    }
  }
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