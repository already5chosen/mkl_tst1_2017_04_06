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

extern "C" void avx256_noncblas_sgemm_ns2x4(
 int M, int N, int K,
 float alpha,
 const float *A, int lda,
 const float *B, int ldb,
 float beta,
 float *C, int ldc);


#ifdef _MSC_VER
#define strncasecmp _strnicmp
#endif

int main(int argz, char** argv)
{
  int M0 = 100,  M1 = 100;
  int N0 = 300,  N1 = 300;
  int K0 = 1000, K1 = 1000;
  int deltaM = 1;
  int deltaN = 1;
  int deltaK = 1;
  float alpha = 1;
  float beta  = 0;
  const int NITER_MIN = 11;
  const int MIN_WORKING_SET_SZ = 32 * 1000 * 1000;

  for (int arg_i = 1; arg_i < argz; ++arg_i) {
    char* arg = argv[arg_i];
    static const char* prefTab[] = {
      "alpha", "beta", "M", "N", "K"
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

  printf("# Running SGEMM with M=%d:%d:%d, N=%d:%d:%d, K=%d:%d:%d, alpha=%f, beta=%f\n",
    M0, deltaM, M1, N0, deltaN, N1, K0, deltaK, K1, alpha, beta);

  int sz = (M1*N1 + M1*K1 + N1*K1)*sizeof(float);
  int nIter = std::max(MIN_WORKING_SET_SZ/sz, NITER_MIN);

  size_t A_SZ = ((M1*K1*sizeof(float)-1)/64 + 1)*(64/sizeof(float));
  size_t B_SZ = ((K1*N1*sizeof(float)-1)/64 + 1)*(64/sizeof(float));
  size_t C_SZ = ((M1*N1*sizeof(float)-1)/64 + 1)*(64/sizeof(float));
  char* bufAlloc = new char [(A_SZ+B_SZ+C_SZ*2)*nIter*sizeof(float)+64];
  uintptr_t bufAdj = (0-(uintptr_t)(bufAlloc)) % 64;
  float* buf = (float*)(bufAlloc+bufAdj);

  size_t AB_SZxN = (A_SZ + B_SZ)*nIter;
  size_t C_SZxN  = C_SZ*nIter;
  float* AB   = buf;
  float* C    = AB + AB_SZxN;
  float* srcC = C + C_SZxN;

  std::mt19937_64 rndGen;
  std::uniform_real_distribution<float> rndDistr(-1.0f, 1.0f);
  auto rndFunc = std::bind ( rndDistr, std::ref(rndGen) );
  for (unsigned i = 0; i < AB_SZxN; ++i)
    AB[i] = rndFunc();
  for (unsigned i = 0; i < C_SZxN; ++i)
    srcC[i] = rndFunc();


  float dummy = 0;
  for (int m = M0; m <= M1; m += deltaM) {
    for (int n = N0; n <= N1; n += deltaN) {
      for (int k = K0; k <= K1; k += deltaK) {
        ::Sleep(10);
        int a_sz = ((m*k*sizeof(float)-1)/64 + 1)*(64/sizeof(float));
        int b_sz = ((k*n*sizeof(float)-1)/64 + 1)*(64/sizeof(float));
        int c_sz = ((m*n*sizeof(float)-1)/64 + 1)*(64/sizeof(float));
        int nIt = std::min(AB_SZxN/(a_sz+b_sz), C_SZxN/c_sz);
        memcpy(C, srcC, c_sz*nIt*sizeof(float));
        std::vector<uint64_t> dt(nIt);
        float* A = AB;
        float* B = &AB[a_sz*nIt];
        for (int it = 0; it < nIt; ++it) {
          uint64_t t0 = __rdtsc();
          avx256_noncblas_sgemm_ns2x4(
            m, n, k, alpha,
            &A[a_sz*it], k,
            &B[b_sz*it], n,
            beta,
            &C[c_sz*it], n);
          uint64_t t1 = __rdtsc();
          dt[it] = t1 - t0;
        }
        for (int i = 0; i < c_sz*nIt; ++i)
          dummy += C[i];
        std::nth_element(dt.begin(), dt.begin()+nIt/2, dt.begin()+nIt);
        printf("%4d %4d %4d %.0f\n", m, n, k, double(dt[nIt/2]));
        fflush(stdout);
      }
    }
  }

  if (argz==100)
    printf("%f\n", dummy);

  delete [] bufAlloc;
  return 0;
}

extern "C" {
uint64_t dbg_tt;
}