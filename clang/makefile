# CC  = /mingw64/bin/clang-cl
# CPP = /mingw64/bin/clang-cl
CC  = clang-cl
CPP = /mingw64/bin/clang-cl
CPPOPT     = -Wall -mavx -O2
CPPOPT_AVX = -Wall -mavx -O1
CPPOPT_FMA = -Wall -mavx -O1 -mfma
COPT_AVX   = -Wall -mavx -O1
COPT_FMA   = -Wall -mavx -O1 -mfma

avx%.o : ../avx%.cpp
	$(CPP) -c $(CPPOPT_AVX) $< -o $@
	g++ -MM ${CPPOPT_AVX} -MF $(patsubst %.o,%.d,$@) $<

fma%.o : ../fma%.cpp
	$(CPP) -c $(CPPOPT_FMA) $< -o $@
	g++ -MM ${CPPOPT_FMA} -MF $(patsubst %.o,%.d,$@) $<

fma%.o : ../fma%.c
	$(CC) -c $(COPT_FMA) $< -o $@
	gcc -MM ${COPT_FMA} -MF $(patsubst %.o,%.d,$@) $<

avx%.o : ../avx%.c
	$(CC) -c $(COPT_AVX) $< -o $@
	gcc -MM ${COPT_AVX} -MF $(patsubst %.o,%.d,$@) $<

%.o : ../%.cpp
	$(CPP) -c $(CPPOPT) $< -o $@
	g++ -MM ${CPPOPT} -MF $(patsubst %.o,%.d,$@) $<

%.s : %.o
	objdump -d -M,intel  $< > $@

ALL_EXE :=
ALL_EXE += tst1.exe
#ALL_EXE += /d/share/mkl/mkl_tst1_g.exe
all: ${ALL_EXE}


OBJLIST :=
OBJLIST += tst1.o
OBJLIST += ref_noncblas_sgemm.o

OBJLIST_AVX :=
OBJLIST_AVX += avxscalar_noncblas_sgemm.o
OBJLIST_AVX += avx128_noncblas_sgemm.o
OBJLIST_AVX += avx256_noncblas_sgemm.o
OBJLIST_AVX += avx128_noncblas_sgemm_m.o
OBJLIST_AVX += avx256_noncblas_sgemm_m.o
OBJLIST_AVX += avx256_noncblas_sgemm_n.o
OBJLIST_AVX += avx256_noncblas_sgemm_n4.o
OBJLIST_AVX += avx256_noncblas_sgemm_ns.o
OBJLIST_AVX += avx256_noncblas_sgemm_ns4.o
OBJLIST_AVX += avx256_noncblas_sgemm_ns4cc.o
OBJLIST_AVX += avx256_noncblas_sgemm_p.o
OBJLIST_AVX += avx256_noncblas_sgemm_a.o
OBJLIST_AVX += avx256saxpy5x2_p.o
OBJLIST_AVX += avx256saxpy4x2_p.o
OBJLIST_AVX += avx256saxpy3x2_p.o
OBJLIST_AVX += avx256saxpy2x2_p.o
OBJLIST_AVX += avx256saxpy1x2_p.o

OBJLIST_FMA :=
OBJLIST_FMA += fma128_noncblas_sgemm.o
OBJLIST_FMA += fma256_noncblas_sgemm.o
OBJLIST_FMA += fma256_noncblas_sgemm_4x3.o
OBJLIST_FMA += fma256_noncblas_sgemm_3x4.o
OBJLIST_FMA += fma256_noncblas_sgemm_4x2.o
OBJLIST_FMA += fma256_noncblas_sgemm_5x2.o
OBJLIST_FMA += fma128_noncblas_sgemm_m.o
OBJLIST_FMA += fma256_noncblas_sgemm_m.o
OBJLIST_FMA += fma256_noncblas_sgemm_n.o
OBJLIST_FMA += fma128_noncblas_sgemm_n.o
OBJLIST_FMA += fma256_noncblas_sgemm_n1.o
OBJLIST_FMA += fma256_noncblas_sgemm_o.o
OBJLIST_FMA += fma256_noncblas_sgemm_p.o


-include $(OBJLIST:.o=.d)
-include $(OBJLIST_AVX:.o=.d)
-include $(OBJLIST_FMA:.o=.d)

tst1.exe : $(OBJLIST) $(OBJLIST_AVX) $(OBJLIST_FMA)
	${CPP} $+ -o$@

/d/share/mkl/mkl_tst1_g.exe : tst1.exe
	cp tst1.exe /d/share/mkl/mkl_tst1_g.exe

clean:
	rm *.o *.d $(ALL_EXE)