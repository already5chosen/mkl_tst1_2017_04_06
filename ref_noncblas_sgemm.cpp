#include <vector>

void ref_noncblas_sgemm(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc)
{
  // Use daxpy method, because, somewhat counterintuitevely, 
  // for big (but not VERY big) matrices it is least slow of all naive methods
  std::vector<double> acc(N);
  for (int m = 0; m < M; A += lda, C += ldc, ++m) {
    // process A and C matrices line by line
    if (beta != 0) {
      for (int n = 0; n < N; ++n)
        acc[n] = double(C[n])*beta;
    } else {
      for (int n = 0; n < N; ++n)
        acc[n] = 0.0;
    }
    const float *bRow = B;
    for (int k = 0; k < K; bRow += ldb, ++k) {
      double ax = double(A[k])*alpha;
      for (int n = 0; n < N; ++n)
        acc[n] += bRow[n]*ax;
    }
    for (int n = 0; n < N; ++n)
      C[n] = float(acc[n]);
  }
}
