#if defined(__GNUC__) || defined(__clang__)
#define NO_MKL
#endif

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <vector>
#include <random>
#include <functional>           // for std::bind
#include <algorithm>
#include <windows.h>
#ifndef NO_MKL
#include "mkl_cblas.h"
#endif
#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif


void ref_noncblas_sgemm(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

void scalar_noncblas_sgemm(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

void avx128_noncblas_sgemm(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

void avx128_noncblas_sgemm_m(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

void fma128_noncblas_sgemm(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

void fma128_noncblas_sgemm_m(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void fma128_noncblas_sgemm_n5(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void fma128_noncblas_sgemm_n5_tune(int m_step, int k_step);

extern "C" void avx256_noncblas_sgemm(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void avx256_noncblas_sgemm_a(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void avx256_noncblas_sgemm_m(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void avx256_noncblas_sgemm_n5(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void avx256_noncblas_sgemm_n4(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void avx256_noncblas_sgemm_ns5(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void avx256_noncblas_sgemm_np5(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void avx256_noncblas_sgemm_ns5r(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void avx256_noncblas_sgemm_ns4(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void avx256_noncblas_sgemm_np4(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void avx256_noncblas_sgemm_ns4cc(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void avx256_noncblas_sgemm_p(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void fma256_noncblas_sgemm_p(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

void fma256_noncblas_sgemm(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

void fma256_noncblas_sgemm_m(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void fma256_noncblas_sgemm_n5(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void fma256_noncblas_sgemm_n5_tune(int m_step, int k_step);

extern "C" void fma256_noncblas_sgemm_nt5(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void fma256_noncblas_sgemm_ns5(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void fma256_noncblas_sgemm_np5(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void fma256_noncblas_sgemm_n4(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void fma256_noncblas_sgemm_ns4(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void fma256_noncblas_sgemm_np4(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void fma256_noncblas_sgemm_n4x3(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void fma256_noncblas_sgemm_ns4x3(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void fma256_noncblas_sgemm_ns3x3(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void fma256_noncblas_sgemm_ns3x4(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void fma256_noncblas_sgemm_ns2x4(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void fma256_noncblas_sgemm_ns2x5(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void avx256_noncblas_sgemm_ns2x4(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void fma256_noncblas_sgemm_ns4x3orig(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void fma256_noncblas_sgemm_n1(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void fma256_noncblas_sgemm_n1_tune(int m_step, int k_step);

extern "C" void fma256_noncblas_sgemm_o(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

extern "C" void fma256_noncblas_sgemm_o_tune(int m_step, int k_step);

void fma256_noncblas_sgemm_3x4(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

void fma256_noncblas_sgemm_4x3(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

void fma256_noncblas_sgemm_4x2(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

void fma256_noncblas_sgemm_5x2(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

#ifndef NO_MKL
// adapt MKL cblas_sgemm to my 'noncblas' calling order
static void MKL_noncblas_sgemm(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc)
{
  cblas_sgemm(
    CblasRowMajor, CblasNoTrans, CblasNoTrans
    , M, N, K
    , alpha
    , A, lda
    , B, ldb
    , beta
    , C, ldc);
}
#endif

static void test_noncblas_sgemm(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc,
 int nIter_meas,
 int nIter_check,
 const float *srcC,
void (*uut)(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc)
 );

#ifdef _MSC_VER
#define strncasecmp _strnicmp
#endif

bool IsFMA3Supported()
{
#if defined(__GNUC__) || defined(__clang__)
  int EAX, EBX, ECX, EDX;
  __cpuid (1, EAX, EBX, ECX, EDX);
  //printf("CPU features: %08x:%08x:%08x:%08x\n", EAX, EBX, ECX, EDX);
#else
  int cpuInfo[4];
  __cpuid(cpuInfo, 1); //  EAX, EBX, ECX, and EDX
  //printf("CPU features: %08x:%08x:%08x:%08x\n", cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
  int ECX = cpuInfo[2];
#endif
  return (ECX & (1 << 12)) != 0; // check for a presence of FMA3 extension
}

static void explore_mk(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc,
 int nIter,
 const float *srcC
);

int main(int argz, char** argv)
{
  int M = 100;
  int N = 300;
  int K = 1000;
  float alpha = 1;
  float beta  = 0;
  int lda = 0;
  int ldb = 0;
  int ldc = 0;
  bool explore = false;
  int nIter_check = 11;

  for (int arg_i = 1; arg_i < argz; ++arg_i) {
    char* arg = argv[arg_i];
    static const char* prefTab[] = {
      "alpha", "beta", "M", "N", "K", "lda", "ldb", "ldc", "cn"
    };
    const int prefTabLen = sizeof(prefTab)/sizeof(prefTab[0]);
    for (int pref_i = 0; pref_i < prefTabLen; ++pref_i) {
      const char* pref = prefTab[pref_i];
      size_t preflen = strlen(pref);
      if (arg[0]=='x') explore=true; else
      if ( strncasecmp(pref, arg, preflen)==0 && arg[preflen]=='=') {
        if (pref_i < 2) {
          // floating point arguments
          char* endp;
          double val = strtod(&arg[preflen+1], &endp);
          if (endp==&arg[preflen+1]) {
            fprintf(stderr, "Bad parameter '%s'. '%s' is not a number.\n", arg, &arg[preflen+1]);
            return 1;
          }
          switch (pref_i) {
            case 0: alpha = float(val); break;
            case 1: beta = float(val);  break;
            default:break;
          }
        } else {
          // integer arguments
          char* endp;
          long val = strtol(&arg[preflen+1], &endp, 0);
          if (endp==&arg[preflen+1] || val <= 0) {
            fprintf(stderr, "Bad parameter '%s'. '%s' is not a positive number.\n", arg, &arg[preflen+1]);
            return 1;
          }
          switch (pref_i) {
            case 2: M = val; break;
            case 3: N = val; break;
            case 4: K = val; break;
            case 5: lda = val; break;
            case 6: ldb = val; break;
            case 7: ldc = val; break;
            case 8: nIter_check = val; break;
            default:break;
          }
        }
        goto next_arg;
      }
    }
    next_arg:;
  }

  if (lda == 0) lda = K;
  if (ldb == 0) ldb = N;
  if (ldc == 0) ldc = N;

  if (lda < K) {
    fprintf(stderr, "Bad parameter lda=%d. Should be greater or equal to K=%d\n", lda, K);
    return 1;
  }
  if (ldb < N) {
    fprintf(stderr, "Bad parameter ldb=%d. Should be greater or equal to N=%d\n", ldb, N);
    return 1;
  }
  if (ldc < N) {
    fprintf(stderr, "Bad parameter ldc=%d. Should be greater or equal to N=%d\n", ldc, N);
    return 1;
  }

  printf("Running SGEMM with M=%d, N=%d, K=%d, alpha=%f, lda=%d, ldb=%d, beta=%f, ldc=%d\n",
    M, N, K, alpha, lda, ldb, beta, ldc);

  nIter_check = nIter_check < 1 ? 1 : nIter_check;
  int nIter_meas = nIter_check < 11 ? 11 : nIter_check;

  const int MIN_WORKING_SET_SZ = 32 * 1000 * 1000;
  int sz = (M*N + M*K + N*K)*sizeof(float);
  if (sz * nIter_meas < MIN_WORKING_SET_SZ)
    nIter_meas = MIN_WORKING_SET_SZ / (sz * 2) * 2 + 1;


  std::vector<float> A(nIter_meas*M*lda);
  std::vector<float> B(nIter_meas*K*ldb);
  std::vector<float> C(nIter_meas*M*ldc);
  std::vector<float> srcC(nIter_meas*M*ldc);

  std::mt19937_64 rndGen;
  std::uniform_real_distribution<float> rndDistr(-1.0f, 1.0f);
  auto rndFunc = std::bind ( rndDistr, std::ref(rndGen) );
  for (int i = 0; i < nIter_meas*M*lda; ++i)
    A[i] = rndFunc();
  for (int i = 0; i < nIter_meas*K*ldb; ++i)
    B[i] = rndFunc();
  for (int i = 0; i < nIter_meas*M*ldc; ++i)
    srcC[i] = rndFunc();

  if (explore) {
    explore_mk(M, N, K, alpha
    , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
    , nIter_meas, &srcC.at(0));
    return 0;
  }

  bool hasFma3 = IsFMA3Supported();

#if 0
  printf("Testing my scalar hack...\n");
  test_noncblas_sgemm(M, N, K, alpha
    , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
    , nIter_meas, nIter_check, &srcC.at(0),
    scalar_noncblas_sgemm);
#endif

#if 0
#ifndef NO_MKL
  printf("Testing Intel MKL...\n");
  test_noncblas_sgemm(M, N, K, alpha
    , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
    , nIter_meas, nIter_check, &srcC.at(0),
    MKL_noncblas_sgemm);
#endif
#endif

#if 0
  printf("Testing my 128-bit AVX hack (5x2 inner loop with 4x2 helper)...\n");
  test_noncblas_sgemm(M, N, K, alpha
    , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
    , nIter_meas, nIter_check, &srcC.at(0),
    avx128_noncblas_sgemm_m);

  printf("Testing my 128-bit AVX hack...\n");
  test_noncblas_sgemm(M, N, K, alpha
    , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
    , nIter_meas, nIter_check, &srcC.at(0),
    avx128_noncblas_sgemm);
#endif

#if 0
  if (hasFma3) {
    printf("Testing my 128-bit FMA hack (2x5 inner loop with 1x5 helper)...\n");
    test_noncblas_sgemm(M, N, K, alpha
      , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
      , nIter_meas, nIter_check, &srcC.at(0),
      fma128_noncblas_sgemm_n5);
  }
  if (hasFma3) {
    printf("Testing my 128-bit FMA hack (2x5 inner loop with 1x5 helper)...\n");
    test_noncblas_sgemm(M, N, K, alpha
      , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
      , nIter_meas, nIter_check, &srcC.at(0),
      fma128_noncblas_sgemm_n5);
  }
#endif

#if 0
  if (hasFma3) {
    printf("Testing my 128-bit FMA hack...\n");
    test_noncblas_sgemm(M, N, K, alpha
      , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
      , nIter_meas, nIter_check, &srcC.at(0),
      fma128_noncblas_sgemm);
  }
#endif

#if 0
  if (hasFma3) {
    printf("Testing my 128-bit FMA hack (5x2 inner loop with 4x2 helper)...\n");
    test_noncblas_sgemm(M, N, K, alpha
      , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
      , nIter_meas, nIter_check, &srcC.at(0),
      fma128_noncblas_sgemm_m);
  }
#endif

#if 0
  printf("Testing my 256-bit AVX hack (2x5 inner loop with 1x5 helper)...\n");
  ::Sleep(100);
  test_noncblas_sgemm(M, N, K, alpha
    , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
    , nIter_meas, nIter_check, &srcC.at(0),
    avx256_noncblas_sgemm_n5);
#endif

#if 0
  printf("Testing my 256-bit AVX hack (mix of different cores)...\n");
  ::Sleep(100);
  test_noncblas_sgemm(M, N, K, alpha
    , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
    , nIter_meas, nIter_check, &srcC.at(0),
    avx256_noncblas_sgemm_a);
#endif

#if 0
  printf("Testing my 256-bit AVX hack (2x5 inner loop with 1x5 helper, no copy of A)...\n");
  ::Sleep(100);
  test_noncblas_sgemm(M, N, K, alpha
    , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
    , nIter_meas, nIter_check, &srcC.at(0),
    avx256_noncblas_sgemm_ns5);
#endif

#if 0
  printf("Testing my 256-bit AVX hack (2x5 inner loop with 1x5 helper, no copy of A or B)...\n");
  ::Sleep(100);
  test_noncblas_sgemm(M, N, K, alpha
    , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
    , nIter_meas, nIter_check, &srcC.at(0),
    avx256_noncblas_sgemm_np5);
#endif

#if 0
  printf("Testing my 256-bit AVX hack (2x5 inner loop with 1x5 helper, no copy of A, inner loop not unrolled)...\n");
  ::Sleep(100);
  test_noncblas_sgemm(M, N, K, alpha
    , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
    , nIter_meas, nIter_check, &srcC.at(0),
    avx256_noncblas_sgemm_ns5r);
#endif

#if 0
  printf("Testing my 256-bit AVX hack (2x4 inner loop with 1x4 helper)...\n");
  ::Sleep(100);
  test_noncblas_sgemm(M, N, K, alpha
    , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
    , nIter_meas, nIter_check, &srcC.at(0),
    avx256_noncblas_sgemm_n4);
#endif

#if 0
  printf("Testing my 256-bit AVX hack (2x4 inner loop with 1x4 helper, no copy of A)...\n");
  ::Sleep(100);
  test_noncblas_sgemm(M, N, K, alpha
    , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
    , nIter_meas, nIter_check, &srcC.at(0),
    avx256_noncblas_sgemm_ns4);
#endif

#if 0
  printf("Testing my 256-bit AVX hack (2x4 inner loop with 1x4 helper, no copy of A or B)...\n");
  ::Sleep(100);
  test_noncblas_sgemm(M, N, K, alpha
    , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
    , nIter_meas, nIter_check, &srcC.at(0),
    avx256_noncblas_sgemm_np4);
#endif

#if 0
  printf("Testing my 256-bit AVX hack (2x4 inner loop with 1x4 helper, no copy of A, buffered cc)...\n");
  ::Sleep(100);
  test_noncblas_sgemm(M, N, K, alpha
    , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
    , nIter_meas, nIter_check, &srcC.at(0),
    avx256_noncblas_sgemm_ns4cc);
#endif

#if 0
  printf("Testing my 256-bit AVX hack (2x5 saxpy inner loop)...\n");
  ::Sleep(100);
  test_noncblas_sgemm(M, N, K, alpha
    , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
    , nIter_meas, nIter_check, &srcC.at(0),
    avx256_noncblas_sgemm_p);
#endif

#if 0
  printf("Testing my 256-bit AVX hack (5x2 inner loop with 4x2 helper)...\n");
  test_noncblas_sgemm(M, N, K, alpha
    , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
    , nIter_meas, nIter_check, &srcC.at(0),
    avx256_noncblas_sgemm_m);
#endif

#if 0
  printf("Testing my 256-bit AVX hack (3x2 inner loop)...\n");
  test_noncblas_sgemm(M, N, K, alpha
    , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
    , nIter_meas, nIter_check, &srcC.at(0),
    avx256_noncblas_sgemm);
#endif

#if 0
  if (hasFma3) {
    printf("Testing my 256-bit FMA hack (5x2 inner loop)...\n");
    test_noncblas_sgemm(M, N, K, alpha
      , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
      , nIter, &srcC.at(0),
      fma256_noncblas_sgemm);
  }
#endif
#if 0
  if (hasFma3) {
    printf("Testing my 256-bit FMA hack (2x3 inner loop with 1x3 helper)...\n");
    printf("%p %p %p\n", &A.at(0), &B.at(0), &C.at(0));
    test_noncblas_sgemm(M, N, K, alpha
      , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
      , nIter_meas, nIter_check, &srcC.at(0),
      fma256_noncblas_sgemm_o);
  }
  if (hasFma3) {
    printf("Testing my 256-bit FMA hack (2x5 inner loop with 1x5 helper, simplified)...\n");
    test_noncblas_sgemm(M, N, K, alpha
      , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
      , nIter_meas, nIter_check, &srcC.at(0),
      fma256_noncblas_sgemm_n1);
  }
#endif

#if 1
  printf("\n");
  if (hasFma3) {
    ::Sleep(100);
    printf("Testing my 256-bit FMA hack (2x5 inner loop with 1x5 helper)...\n");
    test_noncblas_sgemm(M, N, K, alpha
      , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
      , nIter_meas, nIter_check, &srcC.at(0),
      fma256_noncblas_sgemm_n5);
  }
#endif

#if 0
  if (hasFma3) {
    printf("Testing my 256-bit FMA hack (2x5 inner loop with 1x5 helper, no copy of A)...\n");
    ::Sleep(100);
    test_noncblas_sgemm(M, N, K, alpha
      , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
      , nIter_meas, nIter_check, &srcC.at(0),
      fma256_noncblas_sgemm_ns5);
  }
#endif

#if 0
  printf("Testing my 256-bit FMA hack (2x5 inner loop with 1x5 helper, no copy of A or B)...\n");
  ::Sleep(100);
  test_noncblas_sgemm(M, N, K, alpha
    , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
    , nIter_meas, nIter_check, &srcC.at(0),
    fma256_noncblas_sgemm_np5);
#endif

#if 0
  if (hasFma3) {
    printf("Testing my 256-bit FMA hack (2x4 inner loop with 1x4 helper)...\n");
    ::Sleep(100);
    test_noncblas_sgemm(M, N, K, alpha
      , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
      , nIter_meas, nIter_check, &srcC.at(0),
      fma256_noncblas_sgemm_n4);
  }
#endif

#if 0
  if (hasFma3) {
    printf("Testing my 256-bit FMA hack (2x4 inner loop with 1x4 helper, no copy of A)...\n");
    ::Sleep(100);
    test_noncblas_sgemm(M, N, K, alpha
      , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
      , nIter_meas, nIter_check, &srcC.at(0),
      fma256_noncblas_sgemm_ns4);
  }
#endif

#if 0
  if (hasFma3) {
    printf("Testing my 256-bit FMA hack (4 rows X 3*8 columns inner loop)...\n");
    ::Sleep(100);
    test_noncblas_sgemm(M, N, K, alpha
      , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
      , nIter_meas, nIter_check, &srcC.at(0),
      fma256_noncblas_sgemm_n4x3);
  }
#endif

#if 1
  if (hasFma3) {
    printf("Testing my 256-bit FMA hack (4 rows X 3*8 columns inner loop, no copy of A)...\n");
    ::Sleep(100);
    test_noncblas_sgemm(M, N, K, alpha
      , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
      , nIter_meas, nIter_check, &srcC.at(0),
      fma256_noncblas_sgemm_ns4x3);
  }
#endif

#if 0
  if (hasFma3) {
    printf("Testing my 256-bit FMA hack (3 rows X 3*8 columns inner loop, no copy of A)...\n");
    ::Sleep(100);
    test_noncblas_sgemm(M, N, K, alpha
      , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
      , nIter_meas, nIter_check, &srcC.at(0),
      fma256_noncblas_sgemm_ns3x3);
  }
#endif

#if 0
  if (hasFma3) {
    printf("Testing my 256-bit FMA hack (3 rows X 4*8 columns inner loop, no copy of A)...\n");
    ::Sleep(100);
    test_noncblas_sgemm(M, N, K, alpha
      , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
      , nIter_meas, nIter_check, &srcC.at(0),
      fma256_noncblas_sgemm_ns3x4);
  }
#endif

#if 0
  if (hasFma3) {
    printf("Testing my 256-bit FMA hack (2 rows X 4*8 columns inner loop, no copy of A)...\n");
    ::Sleep(100);
    test_noncblas_sgemm(M, N, K, alpha
      , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
      , nIter_meas, nIter_check, &srcC.at(0),
      fma256_noncblas_sgemm_ns2x4);
  }
#endif

#if 1
  if (hasFma3) {
    printf("Testing my 256-bit FMA hack (2 rows X 5*8 columns inner loop, no copy of A)...\n");
    ::Sleep(100);
    test_noncblas_sgemm(M, N, K, alpha
      , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
      , nIter_meas, nIter_check, &srcC.at(0),
      fma256_noncblas_sgemm_ns2x5);
  }
#endif

#if 0
    printf("Testing my 256-bit AVX hack (2 rows X 4*8 columns inner loop, no copy of A)...\n");
    ::Sleep(100);
    test_noncblas_sgemm(M, N, K, alpha
      , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
      , nIter_meas, nIter_check, &srcC.at(0),
      avx256_noncblas_sgemm_ns2x4);
#endif

#if 0
  if (hasFma3) {
    printf("Testing my 256-bit FMA hack (4 rows X 3*8 columns inner loop, no copy of A) first variant...\n");
    ::Sleep(100);
    test_noncblas_sgemm(M, N, K, alpha
      , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
      , nIter_meas, nIter_check, &srcC.at(0),
      fma256_noncblas_sgemm_ns4x3orig);
  }
#endif

#if 0
  if (hasFma3) {
    printf("Testing my 256-bit FMA hack (4 rows X 3*8 columns inner loop, no copy of A) first version ...\n");
    ::Sleep(100);
    test_noncblas_sgemm(M, N, K, alpha
      , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
      , nIter_meas, nIter_check, &srcC.at(0),
      fma256_noncblas_sgemm_ns4x3fma256_noncblas_sgemm_ns4x3orig);
  }
#endif

#if 0
  printf("Testing my 256-bit FMA hack (2x4 inner loop with 1x4 helper, no copy of A or B)...\n");
  ::Sleep(100);
  test_noncblas_sgemm(M, N, K, alpha
    , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
    , nIter_meas, nIter_check, &srcC.at(0),
    fma256_noncblas_sgemm_np4);
#endif

#if 0
  if (hasFma3) {
    printf("Testing my 256-bit FMA hack (2x5 saxpy inner loop)...\n");
    ::Sleep(100);
    test_noncblas_sgemm(M, N, K, alpha
      , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
      , nIter_meas, nIter_check, &srcC.at(0),
      fma256_noncblas_sgemm_p);
    }
#endif


#if 0
  if (hasFma3) {
    printf("Testing my 256-bit FMA hack (5x2 inner loop with 4x2 helper)...\n");
    test_noncblas_sgemm(M, N, K, alpha
      , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
      , nIter_meas, nIter_check, &srcC.at(0),
      fma256_noncblas_sgemm_m);
  }
#endif

#if 0
  if (hasFma3) {
    printf("Testing my 256-bit FMA hack (5x2 inner loop, Enh)...\n");
    test_noncblas_sgemm(M, N, K, alpha
      , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
      , nIter_meas, nIter_check, &srcC.at(0),
      fma256_noncblas_sgemm_5x2);
  }
#endif

#if 0
  if (hasFma3) {
    printf("Testing my 256-bit FMA hack (3x4 inner loop)...\n");
    test_noncblas_sgemm(M, N, K, alpha
      , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
      , nIter_meas, nIter_check, &srcC.at(0),
      fma256_noncblas_sgemm_3x4);
  }
#endif

#if 0
  if (hasFma3) {
    printf("Testing my 256-bit FMA hack (4x3 inner loop)...\n");
    test_noncblas_sgemm(M, N, K, alpha
      , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
      , nIter_meas, nIter_check, &srcC.at(0),
      fma256_noncblas_sgemm_4x3);
  }
#endif

#if 0
  if (hasFma3) {
    printf("Testing my 256-bit FMA hack (4x2 inner loop)...\n");
    test_noncblas_sgemm(M, N, K, alpha
      , &A.at(0), lda, &B.at(0), ldb, beta, &C.at(0), ldc
      , nIter_meas, nIter_check, &srcC.at(0),
      fma256_noncblas_sgemm_4x2);
  }
#endif
  return 0;
}

static void cmp_results(
 int M, int N,
 const float *ref,
 const float *res,
 int ld)
{
  double maxErr = 0;
  double s2Err = 0;
  double s1Ref = 0;
  double s2Ref = 0;
  int maxI = 0;
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      double refV = ref[m*ld+n];
      double resV = res[m*ld+n];
      double err  = resV - refV;
      if (maxErr < fabs(err)) {
        maxErr = fabs(err);
        maxI = m*ld+n;
      }
      s2Err += err*err;
      s1Ref += refV;
      s2Ref += refV*refV;
    }
  }
  double stdErr = sqrt(s2Err / (M*N));
  double stdRef = sqrt(s2Ref*(M*N) - s1Ref*s1Ref)/((M*N));
  printf("%.3e/%.3e=%.3e. %.3e at [%3d,%3d] %18.10e vs %18.10e %s\n"
    , stdErr, stdRef, stdErr/stdRef
    , maxErr, maxI/ld, maxI%ld
    , double(ref[maxI]), double(res[maxI])
    , maxErr > stdRef*1e-5 ? "FAIL !!!" : (maxErr > stdRef*3e-5 || stdErr > stdRef*1e-6 ? "Sucks !" : "")
    );
}

static uint64_t qpc() {
  LARGE_INTEGER r;
  ::QueryPerformanceCounter(&r);
  return r.QuadPart;
}

extern "C" {
uint64_t dbg_tt;
}
static void test_noncblas_sgemm(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc,
 int nIter_meas,
 int nIter_check,
 const float *srcC,
void (*uut)(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc)
 )
{
  LARGE_INTEGER pfr;
  ::QueryPerformanceFrequency(&pfr);
  for (int i = 0; i < nIter_meas*M*ldc; ++i)
    C[i] = srcC[i];

  std::vector<uint64_t> dt(nIter_meas);
  std::vector<uint64_t> dpc(nIter_meas);
  std::vector<uint64_t> tt0(nIter_meas);
  std::vector<uint64_t> tt1(nIter_meas);
  // std::vector<uint64_t> dx(nIter_meas);
  for (int it = 0; it < nIter_meas; ++it) {
    dbg_tt = 0;
    uint64_t pc0 = qpc();
    uint64_t t0 = __rdtsc();
    uut(
      M, N, K
      , alpha
      , &A[it*M*lda], lda
      , &B[it*K*ldb], ldb
      , beta
      , &C[it*M*ldc], ldc
      );
    uint64_t t1 = __rdtsc();
    uint64_t pc1 = qpc();
    dt[it] = t1-t0;
    dpc[it] = pc1-pc0;
    tt0[it] = dbg_tt;
    tt1[it] = dt[it]-dbg_tt;
    // t0 = __rdtsc();
    // avx32_fadd_latency(A[it*M*lda], B[it*K*ldb], 100000);
    // t1 = __rdtsc();
    // dx[it] = t1-t0;
  }
  // for (int it = 0; it < nIter_meas; ++it)
    // printf(" %.0f", double(tt[it]));
  // printf("\n");
  for (int it = 0; it < nIter_meas; ++it)
    printf(" %.0f", double(dt[it]));
    // printf(" %.0f/%.0f", double(dt[it]), double(dx[it]));
  std::nth_element(tt0.begin(), tt0.begin()+nIter_meas/2, tt0.begin()+nIter_meas);
  std::nth_element(tt1.begin(), tt1.begin()+nIter_meas/2, tt1.begin()+nIter_meas);
  std::nth_element(dt.begin(), dt.begin()+nIter_meas/2, dt.begin()+nIter_meas);
  std::nth_element(dpc.begin(), dpc.begin()+nIter_meas/2, dpc.begin()+nIter_meas);
  printf(":\n med %.0f. %.3f FLOP/clk %.3f GFLOP/clk    %.0f + %.0f = %.0f+\n"
    , double(dt[nIter_meas/2])
    , double(M)*N*K*2/double(dt[nIter_meas/2])
    , pfr.QuadPart*1e-9*double(M)*N*K*2/double(dpc[nIter_meas/2])
    , double(tt0[nIter_meas/2])
    , double(tt1[nIter_meas/2])
    , double(dt[nIter_meas/2])
    );

  std::vector<float> refC(M*ldc);
  for (int it = 0; it < nIter_check; ++it) {
    for (int i = 0; i < M*ldc; ++i)
      refC[i] = srcC[it*M*ldc+i];
    ref_noncblas_sgemm(
      M, N, K
      , alpha
      , &A[it*M*lda], lda
      , &B[it*K*ldb], ldb
      , beta
      , &refC.at(0), ldc
      );
    cmp_results(
      M, N
      , &refC.at(0)
      , &C[it*M*ldc]
      , ldc
      );
  }
}

static void explore_mk(
  int M, int N, int K,
  float alpha,
  const float *A, int lda,
  const float *B, int ldb,
  float beta,
  float *C, int ldc,
  int nIter,
  const float *srcC)
{
#if 0
  //const int m_step0 = 30;
  //const int m_step1 = 300;

  const int m_step0 = 75;
  const int m_step1 = 400;

  double prev = 0;
  double mx = 0;
  int mxm = 0, mxk = 0;
  std::vector<uint64_t> dt(nIter);
  for (int k_step = 64; k_step <= 400; k_step += 4) {
  //for (int k_step = 188; k_step <= 192; k_step += 4) {
    gl_k_step = k_step;
    for (int m_step = m_step0; ; ) {
      gl_m_step = m_step;

      for (int i = 0; i < nIter*M*ldc; ++i)
        C[i] = srcC[i];
      for (int it = 0; it < nIter; ++it) {
        uint64_t t0 = __rdtsc();
        fma256_noncblas_sgemm_n5(
          M, N, K
          , alpha
          , &A[it*M*lda], lda
          , &B[it*K*ldb], ldb
          , beta
          , &C[it*M*ldc], ldc
          );
        uint64_t t1 = __rdtsc();
        dt[it] = t1 - t0;
      }
      std::nth_element(dt.begin(), dt.begin() + nIter / 2, dt.begin() + nIter);
      double FLOP = double(M)*N*K*2/double(dt[nIter/2]);

      bool best = mx < FLOP;
      if (best) {
        mxm = m_step;
        mxk = k_step;
        mx  = FLOP;
      }
      printf("Max %.3f at %3d,%3d %s   k_step=%3d, m_step=%3d, %.3f FLOPs/Hz %c\n"
        , mx, mxk, mxm
        , FLOP < prev*0.990 ? "--" :
          FLOP < prev*0.998 ? "=-" :
          FLOP < prev*1.002 ? "==" :
          FLOP < prev*1.010 ? "=+" :
                              "++"
        ,k_step, m_step, FLOP
        , best ? '^' : ' '
        );
      prev = FLOP;

      if (gl_m_step < M) {
        if (m_step < m_step1)
          m_step += 5;
        else
          m_step = ((M - 1) / 5 + 1) * 5;
      } else {
        break;
      }
    }
  }
#elif 0
  const int m_step0 = 75;
  const int m_step1 = 1500;

  int m_div0 = (M-1)/m_step0 + 1;

  double prev = 0;
  double mx = 0;
  int mxm = 0, mxk = 0;
  std::vector<uint64_t> dt(nIter);
  for (int m_div = m_div0; m_div > 0;) {
    int m_step = ((M - 1) / (5 * m_div) + 1) * 5;
    gl_m_step = m_step;

    for (int i = 0; i < nIter*M*ldc; ++i)
      C[i] = srcC[i];
    for (int it = 0; it < nIter; ++it) {
      uint64_t t0 = __rdtsc();
      fma256_noncblas_sgemm_n5(
        M, N, K
        , alpha
        , &A[it*M*lda], lda
        , &B[it*K*ldb], ldb
        , beta
        , &C[it*M*ldc], ldc
        );
      uint64_t t1 = __rdtsc();
      dt[it] = t1 - t0;
    }
    std::nth_element(dt.begin(), dt.begin() + nIter / 2, dt.begin() + nIter);
    double FLOP = double(M)*N*K * 2 / double(dt[nIter / 2]);

    bool best = mx < FLOP;
    if (best) {
      mxm = m_step;
      mxk = gl_k_step;
      mx = FLOP;
    }
    printf("Max %.3f at %3d,%3d %s   k_step=%3d, m_step=%3d, %.3f FLOPs/Hz %c\n"
      , mx, mxk, mxm
      , FLOP < prev*0.990 ? "--" :
      FLOP < prev*0.998 ? "=-" :
      FLOP < prev*1.002 ? "==" :
      FLOP < prev*1.010 ? "=+" :
      "++"
      , gl_k_step, m_step, FLOP
      , best ? '^' : ' '
      );
    prev = FLOP;

    if (m_step >= M)
      break;

    if (m_step < m_step1)
      m_div -= 1;
    else
      m_div = 1;
  }
#elif 0
  double prev = 0;
  double mx = 0;
  int mxk = 0;
  std::vector<uint64_t> dt(nIter);
  for (int k_step = 32; k_step <= 400 && k_step < K; k_step += 4) {
    gl_k_step = k_step;

    for (int i = 0; i < nIter*M*ldc; ++i)
      C[i] = srcC[i];

    for (int it = 0; it < nIter; ++it) {
      uint64_t t0 = __rdtsc();
      fma256_noncblas_sgemm_n5(
        M, N, K
        , alpha
        , &A[it*M*lda], lda
        , &B[it*K*ldb], ldb
        , beta
        , &C[it*M*ldc], ldc
        );
      uint64_t t1 = __rdtsc();
      dt[it] = t1 - t0;
    }
    std::nth_element(dt.begin(), dt.begin() + nIter / 2, dt.begin() + nIter);
    double FLOP = double(M)*N*K * 2 / double(dt[nIter / 2]);

    bool best = mx < FLOP;
    if (best) {
      mxk = k_step;
      mx = FLOP;
    }
    printf("Max %.3f at %3d %s   k_step=%3d, %.3f FLOPs/Hz %c\n"
      , mx, mxk
      , FLOP < prev*0.990 ? "--" :
      FLOP < prev*0.998 ? "=-" :
      FLOP < prev*1.002 ? "==" :
      FLOP < prev*1.010 ? "=+" :
      "++"
      , k_step, FLOP
      , best ? '^' : ' '
      );
    prev = FLOP;
  }
#elif 0
  const int n_superstep0 = 1;//(N-1)/50+1;
  const int k_step0 = 1;
  const int m_div0 = (M-1)/16+1;

  double prev = 0;
  double mx = 0;
  int mxm = 0, mxk = 0, mxn=0;
  std::vector<uint64_t> dt(nIter);
  gl_n_superstep = -1;
  for (int n_superstepReq = n_superstep0; n_superstepReq <= N; ++n_superstepReq) {
    int n_superstep = n_superstepReq;
    if (n_superstep != N) {
      const int n_step = 32;
      n_superstep = N;
      if (n_superstepReq > n_step && n_superstepReq < N) {
        int n_Nsupersteps = (N * 2 - n_superstepReq) / (2 * n_superstepReq) + 1;
        if (n_Nsupersteps >= 2) {
          int n_Nsteps = (N - 1) / n_step + 1;
          int n_stepsPerSuper = n_Nsteps / n_Nsupersteps;
          if ((n_Nsteps - 1) % n_stepsPerSuper <= (n_Nsteps - 1) % (n_stepsPerSuper + 1))
            n_stepsPerSuper += 1;
          n_superstep = n_stepsPerSuper * n_step;
        }
      }
      if (n_superstep == N)
        continue;
    }

    if (gl_n_superstep == n_superstep)
      continue;

    gl_n_superstep = n_superstep;
    gl_k_step = -1;
    for (int k_step = k_step0; k_step <= K; ++k_step) {
      int k_Nsteps = (K*4-k_step)/(4*k_step) + 1;
      int eff_k_step = k_Nsteps < 2 ? K : ((K-1)/(k_Nsteps*4) + 1) * 4;
      if (eff_k_step == gl_k_step)
        continue;

      gl_k_step = eff_k_step;
      gl_m_step = 0;
        for (int i = 0; i < nIter*M*ldc; ++i)
          C[i] = srcC[i];
        for (int it = 0; it < nIter; ++it) {
          uint64_t t0 = __rdtsc();
          fma256_noncblas_sgemm_n5(
            M, N, K
            , alpha
            , &A[it*M*lda], lda
            , &B[it*K*ldb], ldb
            , beta
            , &C[it*M*ldc], ldc
            );
          uint64_t t1 = __rdtsc();
          dt[it] = t1 - t0;
        }
        std::nth_element(dt.begin(), dt.begin() + nIter / 2, dt.begin() + nIter);
        double FLOP = double(M)*N*K * 2 / double(dt[nIter / 2]);

        bool best = mx < FLOP;
        if (best) {
          mxn = gl_n_superstep;
          mxk = gl_k_step;
          mxm = gl_m_step;
          mx = FLOP;
        }

        const char* dirStr = "==";
        if (FLOP < 0.998*prev || FLOP > 1.002*prev) {
          if (FLOP < 0.990*prev)
            dirStr = "--";
          else if (FLOP < prev)
            dirStr = "=-";
          else if (FLOP < 1.010*prev)
            dirStr = "=+";
          else
            dirStr = "++";
          prev = FLOP;
        }

        printf("%d/%d/%d Max %.3f at %3d,%3d,%3d %s  n_sstep=%3d k_step=%3d, m_step=%3d, %.3f FLOPs/Hz %c\n"
          , M, N, K
          , mx, mxn, mxk, mxm, dirStr
          , gl_n_superstep, gl_k_step, gl_m_step, FLOP
          , best ? '^' : ' '
          );

    }
  }
#elif 0
  const int n_superstep0 = 1;//(N-1)/50+1;
  const int k_step0 = 1;
  const int m_div0 = (M-1)/16+1;

  double prev = 0;
  double mx = 0;
  int mxm = 0, mxk = 0, mxn=0;
  std::vector<uint64_t> dt(nIter);
  gl_n_superstep = -1;
  for (int n_superstepReq = n_superstep0; n_superstepReq <= N; ++n_superstepReq) {
    int n_superstep = n_superstepReq;
    if (n_superstep != N) {
      const int n_step = 32;
      n_superstep = N;
      if (n_superstepReq > n_step && n_superstepReq < N) {
        int n_Nsupersteps = (N * 2 - n_superstepReq) / (2 * n_superstepReq) + 1;
        if (n_Nsupersteps >= 2) {
          int n_Nsteps = (N - 1) / n_step + 1;
          int n_stepsPerSuper = n_Nsteps / n_Nsupersteps;
          if ((n_Nsteps - 1) % n_stepsPerSuper <= (n_Nsteps - 1) % (n_stepsPerSuper + 1))
            n_stepsPerSuper += 1;
          n_superstep = n_stepsPerSuper * n_step;
        }
      }
      if (n_superstep == N)
        continue;
    }

    if (gl_n_superstep == n_superstep)
      continue;

    gl_n_superstep = n_superstep;
    gl_k_step = -1;
    for (int k_step = k_step0; k_step <= K; ++k_step) {
      int k_Nsteps = (K*4-k_step)/(4*k_step) + 1;
      int eff_k_step = k_Nsteps < 2 ? K : ((K-1)/(k_Nsteps*4) + 1) * 4;
      if (eff_k_step == gl_k_step)
        continue;

      gl_k_step = eff_k_step;
      gl_m_step = -1;
      for (int m_div = m_div0; m_div >= 1; --m_div) {
        int m_step =  ((M - 1) / (m_div * 5) + 1) * 5;
        if (gl_m_step == m_step)
          continue;
        gl_m_step = m_step;
        for (int i = 0; i < nIter*M*ldc; ++i)
          C[i] = srcC[i];
        for (int it = 0; it < nIter; ++it) {
          uint64_t t0 = __rdtsc();
          fma256_noncblas_sgemm_n5(
            M, N, K
            , alpha
            , &A[it*M*lda], lda
            , &B[it*K*ldb], ldb
            , beta
            , &C[it*M*ldc], ldc
            );
          uint64_t t1 = __rdtsc();
          dt[it] = t1 - t0;
        }
        std::nth_element(dt.begin(), dt.begin() + nIter / 2, dt.begin() + nIter);
        double FLOP = double(M)*N*K * 2 / double(dt[nIter / 2]);

        bool best = mx < FLOP;
        if (best) {
          mxn = gl_n_superstep;
          mxk = gl_k_step;
          mxm = gl_m_step;
          mx = FLOP;
        }

        const char* dirStr = "==";
        if (FLOP < 0.998*prev || FLOP > 1.002*prev) {
          if (FLOP < 0.990*prev)
            dirStr = "--";
          else if (FLOP < prev)
            dirStr = "=-";
          else if (FLOP < 1.010*prev)
            dirStr = "=+";
          else
            dirStr = "++";
          prev = FLOP;
        }

        printf("%d/%d/%d Max %.3f at %3d,%3d,%3d %s  n_sstep=%3d k_step=%3d, m_step=%3d, %.3f FLOPs/Hz %c\n"
          , M, N, K
          , mx, mxn, mxk, mxm, dirStr
          , gl_n_superstep, gl_k_step, gl_m_step, FLOP
          , best ? '^' : ' '
          );
      }
    }
  }
#elif 1
  const int k_step0 = 10;
  const int m_div0 =  (M - 1) / 10 + 1;
  //const int k_step0 = 60*2;
  //const int m_div0 =  (M - 1) / 70 + 1;

  double prev = 0;
  double mx = 0;
  int mxm = 0, mxk = 0;
  std::vector<uint64_t> dt(nIter);

  int curr_k_step = -1;
  for (int k_step = k_step0; k_step <= K; ++k_step) {
    int k_Nsteps = (K * 4 - k_step) / (4 * k_step) + 1;
    int eff_k_step = k_Nsteps < 2 ? K : ((K - 1) / (k_Nsteps * 4) + 1) * 4;
    if (eff_k_step == curr_k_step)
      continue;

    curr_k_step = eff_k_step;
    int curr_m_step = -1;
    for (int m_div = m_div0; m_div >= 1; --m_div) {
      int m_step = ((M - 1) / (m_div * 5) + 1) * 5;
      if (curr_m_step == m_step)
        continue;
      curr_m_step = m_step;
      fma128_noncblas_sgemm_n5_tune(curr_m_step, curr_k_step);
      for (int i = 0; i < nIter*M*ldc; ++i)
        C[i] = srcC[i];
      for (int it = 0; it < nIter; ++it) {
        uint64_t t0 = __rdtsc();
        fma128_noncblas_sgemm_n5(
          M, N, K
          , alpha
          , &A[it*M*lda], lda
          , &B[it*K*ldb], ldb
          , beta
          , &C[it*M*ldc], ldc
          );
        uint64_t t1 = __rdtsc();
        dt[it] = t1 - t0;
      }
      std::nth_element(dt.begin(), dt.begin() + nIter / 2, dt.begin() + nIter);
      double FLOP = double(M)*N*K * 2 / double(dt[nIter / 2]);

      bool best = mx < FLOP;
      if (best) {
        mxk = curr_k_step;
        mxm = curr_m_step;
        mx = FLOP;
      }

      const char* dirStr = "==";
      if (FLOP < 0.998*prev || FLOP > 1.002*prev) {
        if (FLOP < 0.990*prev)
          dirStr = "--";
        else if (FLOP < prev)
          dirStr = "=-";
        else if (FLOP < 1.010*prev)
          dirStr = "=+";
        else
          dirStr = "++";
        prev = FLOP;
      }

      printf("%d/%d/%d Max %.3f at %3d,%3d %s  k_step=%3d, m_step=%3d, %.3f FLOPs/Hz %c\n"
        , M, N, K
        , mx, mxk, mxm, dirStr
        , curr_k_step, curr_m_step, FLOP
        , best ? '^' : ' '
        );
    }
  }

#else
  //const int k_step0 = 10;
  //const int m_div0 =  (M - 1) / 10 + 1;
  const int k_step0 = 120;
  const int m_div0 =  (M - 1) / 70 + 1;

  double prev = 0;
  double mx = 0;
  int mxm = 0, mxk = 0;
  std::vector<uint64_t> dt(nIter);

  int curr_k_step = -1;
  for (int k_step = k_step0; k_step <= K; ++k_step) {
    int k_Nsteps = (K * 4 - k_step) / (4 * k_step) + 1;
    int eff_k_step = k_Nsteps < 2 ? K : ((K - 1) / (k_Nsteps * 4) + 1) * 4;
    if (eff_k_step == curr_k_step)
      continue;

    curr_k_step = eff_k_step;
    int curr_m_step = -1;
    for (int m_div = m_div0; m_div >= 1; --m_div) {
      int m_step = ((M - 1) / (m_div * 3) + 1) * 3;
      if (curr_m_step == m_step)
        continue;
      curr_m_step = m_step;
      fma256_noncblas_sgemm_o_tune(curr_m_step, curr_k_step);
      for (int i = 0; i < nIter*M*ldc; ++i)
        C[i] = srcC[i];
      for (int it = 0; it < nIter; ++it) {
        uint64_t t0 = __rdtsc();
        fma256_noncblas_sgemm_o(
          M, N, K
          , alpha
          , &A[it*M*lda], lda
          , &B[it*K*ldb], ldb
          , beta
          , &C[it*M*ldc], ldc
          );
        uint64_t t1 = __rdtsc();
        dt[it] = t1 - t0;
      }
      std::nth_element(dt.begin(), dt.begin() + nIter / 2, dt.begin() + nIter);
      double FLOP = double(M)*N*K * 2 / double(dt[nIter / 2]);

      bool best = mx < FLOP;
      if (best) {
        mxk = curr_k_step;
        mxm = curr_m_step;
        mx = FLOP;
      }

      const char* dirStr = "==";
      if (FLOP < 0.998*prev || FLOP > 1.002*prev) {
        if (FLOP < 0.990*prev)
          dirStr = "--";
        else if (FLOP < prev)
          dirStr = "=-";
        else if (FLOP < 1.010*prev)
          dirStr = "=+";
        else
          dirStr = "++";
        prev = FLOP;
      }

      printf("%d/%d/%d Max %.3f at %3d,%3d %s  k_step=%3d, m_step=%3d, %.3f FLOPs/Hz %c\n"
        , M, N, K
        , mx, mxk, mxm, dirStr
        , curr_k_step, curr_m_step, FLOP
        , best ? '^' : ' '
        );
    }
  }

#endif
}

