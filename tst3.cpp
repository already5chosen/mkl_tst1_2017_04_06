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
#else
#include "OpenBLAS/cblas.h"
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

extern "C" {

void fma256_noncblas_sgemm_n5(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

void fma256_noncblas_sgemm_ns5(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

void fma256_noncblas_sgemm_ns2x5(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

void fma256_noncblas_sgemm_nn5x2(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);

}

// adapt OpenBLAS/MKL cblas_sgemm to my 'noncblas' calling order
static void BLAS_noncblas_sgemm(
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

typedef void (*noncblas_sgemm_func_t)(
  int M, int N, int K,
  float alpha,
  const float *A, int lda,
  const float *B, int ldb,
  float beta,
  float *C, int ldc);

struct func_tab_entry_t {
  const char*           name;
  noncblas_sgemm_func_t func;
};

static func_tab_entry_t funcTab[] = {
  { "5x2",  fma256_noncblas_sgemm_n5 },
  { "s5x2", fma256_noncblas_sgemm_ns5 },
  { "s2x5", fma256_noncblas_sgemm_ns2x5 },
  { "nn5x2", fma256_noncblas_sgemm_nn5x2 },
  { "blas", BLAS_noncblas_sgemm },
  {0},
};

static bool cmp_results(
 int M, int N, int K,
 const float *ref,
 const float *res,
 int ld);

#ifdef _MSC_VER
#define strncasecmp _strnicmp
#endif

int main(int argz, char** argv)
{
  int uut_i = 0;
  int M0 = 100,  M1 = 100;
  int N0 = 300,  N1 = 300;
  int K0 = 1000, K1 = 1000;
  int deltaM = 1;
  int deltaN = 1;
  int deltaK = 1;
  float alpha = 1;
  float beta  = 0;
  int dLDA = 0;
  int dLDB = 0;
  int dLDC = 0;
  const int NITER = 3;

  for (int arg_i = 1; arg_i < argz; ++arg_i) {
    char* arg = argv[arg_i];
    static const char* prefTab[] = {
      "alpha", "beta",
      "M", "N", "K",
      "F",
      "dLDA", "dLDB", "dLDC"
    };
    const int prefTabLen = sizeof(prefTab)/sizeof(prefTab[0]);
    for (int pref_i = 0; pref_i < prefTabLen; ++pref_i) {
      const char* pref = prefTab[pref_i];
      size_t preflen = strlen(pref);
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
            case 1: beta  = float(val); break;
            default:break;
          }
        } else if (pref_i == 5) {
          for (int i = 0; funcTab[i].name != 0; ++i) {
            if (strcasecmp(funcTab[i].name, &arg[preflen+1])==0) {
              uut_i = i;
              break;
            }
          }
        } else if (pref_i > 5) {
          // dLDx
          char* p = &arg[preflen+1];
          char* endp;
          long val = strtol(p, &endp, 0);
          if (endp==p || val < 0) {
            if (endp != p)
              *endp = 0;
            fprintf(stderr, "Bad parameter '%s'. '%s' is not a non-negative number.\n", arg, p);
            return 1;
          }
          switch (pref_i) {
            case 6: dLDA = val; break;
            case 7: dLDB = val; break;
            case 8: dLDC = val; break;
            default:break;
          }
        } else {
          // range arguments
          char* p = &arg[preflen+1];
          long values[3];
          int  nVal = 0;
          while (nVal < 3) {
            char* endp;
            long val = strtol(p, &endp, 0);
            if (endp==p || val <= 0) {
              if (endp != p)
                *endp = 0;
              fprintf(stderr, "Bad parameter '%s'. '%s' is not a positive number.\n", arg, p);
              return 1;
            }
            values[nVal] = val;
            ++nVal;
            p = endp;
            if (*p == 0) {
              break;
            }
            if (*p != ':') {
              fprintf(stderr, "Bad parameter '%s'. Stray character '%c'. ':' expected.\n", arg, *p);
              return 1;
            }
            p += 1;
          }
          if (nVal == 1) {
            values[1] = 1;
            values[2] = values[0];
          } else if (nVal == 2) {
            values[2] = values[1];
            values[1] = 1;
          }
          if (values[2] < values[0]) {
            fprintf(stderr, "Bad parameter '%s'. %ld is smaller than %ld.\n", arg, values[2], values[0]);
            return 1;
          }

          switch (pref_i) {
            case 2: M0 = values[0]; M1 = values[2]; deltaM = values[1]; break;
            case 3: N0 = values[0]; N1 = values[2]; deltaN = values[1]; break;
            case 4: K0 = values[0]; K1 = values[2]; deltaK = values[1]; break;
            default:break;
          }
        }
        goto next_arg;
      }
    }
    next_arg:;
  }
  noncblas_sgemm_func_t uut = funcTab[uut_i].func;

  char LDAStr[40]={0}; if (dLDA) sprintf(LDAStr, ", LDA=K+%d", dLDA);
  char LDBStr[40]={0}; if (dLDB) sprintf(LDBStr, ", LDB=N+%d", dLDB);
  char LDCStr[40]={0}; if (dLDC) sprintf(LDCStr, ", LDC=N+%d", dLDC);

  printf("# Running SGEMM %s with M=%d:%d:%d, N=%d:%d:%d, K=%d:%d:%d, alpha=%f, beta=%f%s%s%s\n",
    funcTab[uut_i].name
    , M0, deltaM, M1
    , N0, deltaN, N1
    , K0, deltaK, K1
    , alpha, beta
    , LDAStr, LDBStr, LDCStr
    );

  int nIter = NITER;
  int LDA = K1 + dLDA;
  int LDB = N1 + dLDB;
  int LDC = N1 + dLDC;
  size_t A_SZ = ((M1*LDA*sizeof(float)-1)/64 + 1)*(64/sizeof(float));
  size_t B_SZ = ((K1*LDB*sizeof(float)-1)/64 + 1)*(64/sizeof(float));
  size_t C_SZ = ((M1*LDC*sizeof(float)-1)/64 + 1)*(64/sizeof(float));
  char* bufAlloc = new char [(A_SZ+B_SZ+C_SZ*3)*nIter*sizeof(float)+64];
  uintptr_t bufAdj = (0-(uintptr_t)(bufAlloc)) % 64;
  float* buf = (float*)(bufAlloc+bufAdj);

  size_t AB_SZxN = (A_SZ + B_SZ)*nIter;
  size_t C_SZxN  = C_SZ*nIter;
  float* AB   = buf;
  float* resC = AB + AB_SZxN;
  float* refC = resC + C_SZxN;
  float* srcC = refC + C_SZxN;

  std::mt19937_64 rndGen;
  std::uniform_real_distribution<float> rndDistr(-1.0f, 1.0f);
  auto rndFunc = std::bind ( rndDistr, std::ref(rndGen) );
  for (unsigned i = 0; i < AB_SZxN; ++i)
    AB[i] = rndFunc();
  for (unsigned i = 0; i < C_SZxN; ++i)
    srcC[i] = rndFunc();


  bool ok = true;
  for (int m = M0; m <= M1 && ok; m += deltaM) {
    for (int n = N0; n <= N1 && ok; n += deltaN) {
      int ldb = n + dLDB;
      int ldc = n + dLDC;
      for (int k = K0; k <= K1 && ok; k += deltaK) {
        int lda = k + dLDA;
        int a_sz = ((m*lda*sizeof(float)-1)/64 + 1)*(64/sizeof(float));
        int b_sz = ((k*ldb*sizeof(float)-1)/64 + 1)*(64/sizeof(float));
        int c_sz = ((m*ldc*sizeof(float)-1)/64 + 1)*(64/sizeof(float));

        memcpy(resC, srcC, c_sz*nIter*sizeof(float));
        float* A = AB;
        float* B = &AB[a_sz*nIter];
        for (int it = 0; it < nIter; ++it) {
          uut(
            m, n, k, alpha,
            &A[a_sz*it], lda,
            &B[b_sz*it], ldb,
            beta,
            &resC[c_sz*it], ldc);
        }

        memcpy(refC, srcC, c_sz*nIter*sizeof(float));
        A = AB;
        B = &AB[a_sz*nIter];

        for (int it = 0; it < nIter && ok; ++it) {
          ref_noncblas_sgemm(
            m, n, k, alpha,
            &A[a_sz*it], lda,
            &B[b_sz*it], ldb,
            beta,
            &refC[c_sz*it], ldc);
          ok = cmp_results(
            m, n, k
            , &refC[c_sz*it]
            , &resC[c_sz*it]
            , ldc
            );
        }

        // printf("%4d %4d %4d %.0f\n", m, n, k, double(dtMedArr[N_REP/2]));
        fflush(stdout);
      }
    }
  }

  delete [] bufAlloc;
  return ok ? 0 : 1;
}

extern "C" {
uint64_t dbg_tt;
}

static bool cmp_results(
 int M, int N, int K,
 const float *ref,
 const float *res,
 int ld)
{
  double maxErr = 0;
  double s2Err = 0;
  double s1Ref = 0;
  double s2Ref = 0;
  int maxI = 0;
  int margI = -1;
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
    if (margI < 0)
    for (int n = N; n < ld; ++n) {
      float refV = ref[m*ld+n];
      float resV = res[m*ld+n];
      if (refV != resV) {
        margI = m*ld+n;
        break;
      }
    }
  }
  double stdErr = sqrt(s2Err / (M*N));
  double stdRef = sqrt(s2Ref*(M*N) - s1Ref*s1Ref)/((M*N));
  bool ok = maxErr <= stdRef*1e-5;
  printf("%4d %4d %4d : %.3e/%.3e=%.3e. %.3e at [%3d,%3d] %18.10e vs %18.10e %s"
    , M, N, K
    , stdErr, stdRef, stdErr/stdRef
    , maxErr, maxI/ld, maxI%ld
    , double(ref[maxI]), double(res[maxI])
    , !ok ? "FAIL !!!" : (maxErr > stdRef*3e-5 || stdErr > stdRef*1e-6 ? "Sucks !" : "")
    );
  if (ok && margI >= 0) {
    ok = false;
    printf("Margin mismatch at [%3d,%3d] %18.10e vs %18.10e FAIL !!!"
    , margI/ld, margI%ld
    , double(ref[margI]), double(res[margI])
    );
  }
  printf("\n");
  return ok;
}

