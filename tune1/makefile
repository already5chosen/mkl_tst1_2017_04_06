CC  = gcc
CPP = g++
CPPOPT     = -std=c++11 -Wall -mavx -O2
CPPOPT_AVX = -std=c++11 -Wall -mavx -O1
CPPOPT_FMA = -std=c++11 -Wall -mavx -O1 -mfma
COPT_AVX   = -std=c99 -Wall -mavx -O1 -fno-tree-ter
COPT_FMA   = -std=c99 -Wall -mavx -O1 -mfma

COPT_HIGH_LEVEL_ASM :=
COPT_HIGH_LEVEL_ASM += -fno-auto-inc-dec
COPT_HIGH_LEVEL_ASM += -fno-branch-count-reg
# COPT_HIGH_LEVEL_ASM += -fno-combine-stack-adjustments
COPT_HIGH_LEVEL_ASM += -fno-compare-elim
COPT_HIGH_LEVEL_ASM += -fno-cprop-registers
# COPT_HIGH_LEVEL_ASM += -fno-dce
# COPT_HIGH_LEVEL_ASM += -fno-defer-pop
# COPT_HIGH_LEVEL_ASM += -fno-delayed-branch
# COPT_HIGH_LEVEL_ASM += -fno-dse
# COPT_HIGH_LEVEL_ASM += -fno-forward-propagate
# COPT_HIGH_LEVEL_ASM += -fno-guess-branch-probability
# COPT_HIGH_LEVEL_ASM += -fno-if-conversion2
# COPT_HIGH_LEVEL_ASM += -fno-if-conversion
# COPT_HIGH_LEVEL_ASM += -fno-inline-functions-called-once
# COPT_HIGH_LEVEL_ASM += -fno-ipa-pure-const
# COPT_HIGH_LEVEL_ASM += -fno-ipa-profile
# COPT_HIGH_LEVEL_ASM += -fno-ipa-reference
# COPT_HIGH_LEVEL_ASM += -fno-merge-constants
# COPT_HIGH_LEVEL_ASM += -fno-move-loop-invariants
# COPT_HIGH_LEVEL_ASM += -fno-shrink-wrap
# COPT_HIGH_LEVEL_ASM += -fno-split-wide-types
COPT_HIGH_LEVEL_ASM += -fno-tree-bit-ccp
COPT_HIGH_LEVEL_ASM += -fno-tree-ccp
COPT_HIGH_LEVEL_ASM += -fno-ssa-phiopt
COPT_HIGH_LEVEL_ASM += -fno-tree-ch
COPT_HIGH_LEVEL_ASM += -fno-tree-copy-prop
COPT_HIGH_LEVEL_ASM += -fno-tree-copyrename
COPT_HIGH_LEVEL_ASM += -fno-tree-dce
COPT_HIGH_LEVEL_ASM += -fno-tree-dominator-opts
COPT_HIGH_LEVEL_ASM += -fno-tree-dse
COPT_HIGH_LEVEL_ASM += -fno-tree-forwprop
# COPT_HIGH_LEVEL_ASM += -fno-tree-fre
COPT_HIGH_LEVEL_ASM += -fno-tree-phiprop
COPT_HIGH_LEVEL_ASM += -fno-tree-sink
COPT_HIGH_LEVEL_ASM += -fno-tree-slsr
COPT_HIGH_LEVEL_ASM += -fno-tree-sra
COPT_HIGH_LEVEL_ASM += -fno-tree-pta
COPT_HIGH_LEVEL_ASM += -fno-tree-ter
COPT_HIGH_LEVEL_ASM += -fno-unit-at-a-time

COPT_HIGH_LEVEL_ASM2 :=
COPT_HIGH_LEVEL_ASM2 += -fno-auto-inc-dec
COPT_HIGH_LEVEL_ASM2 += -fno-branch-count-reg
COPT_HIGH_LEVEL_ASM2 += -fno-tree-bit-ccp
COPT_HIGH_LEVEL_ASM2 += -fno-tree-ccp
COPT_HIGH_LEVEL_ASM2 += -fno-ssa-phiopt
COPT_HIGH_LEVEL_ASM2 += -fno-tree-ter

avx%.o : ../avx%.cpp
	$(CPP) -c $(CPPOPT_AVX) $< -o $@
	${CPP} -MM ${CPPOPT_AVX} -MF $(patsubst %.o,%.d,$@) $<

fma%.o : ../fma%.cpp
	$(CPP) -c $(CPPOPT_FMA) $< -o $@
	${CPP} -MM ${CPPOPT_FMA} -MF $(patsubst %.o,%.d,$@) $<

fma%.o : ../fma%.c
	$(CC) -c $(COPT_FMA) $< -o $@
	${CC} -MM ${COPT_FMA} -MF $(patsubst %.o,%.d,$@) $<

avx%.o : ../avx%.c
	$(CC) -c $(COPT_AVX) $< -o $@
	${CC} -MM ${COPT_AVX} -MF $(patsubst %.o,%.d,$@) $<

avx%_p.o : ../avx%_p.c
	$(CC) -c $(COPT_AVX) $(COPT_HIGH_LEVEL_ASM) $< -o $@
	${CC} -MM ${COPT_AVX} $(COPT_HIGH_LEVEL_ASM) -MF $(patsubst %.o,%.d,$@) $<

avx%_ns.o : ../avx%_ns.c
	$(CC) -c $(COPT_AVX) $(COPT_HIGH_LEVEL_ASM2) $< -o $@
	${CC} -MM ${COPT_AVX} -MF $(patsubst %.o,%.d,$@) $<

fma%_p.o : ../fma%_p.c
	$(CC) -c $(COPT_FMA) $(COPT_HIGH_LEVEL_ASM) $< -o $@
	${CC} -MM ${COPT_FMA} -MF $(patsubst %.o,%.d,$@) $<

%.o : ../%.cpp
	$(CPP) -c $(CPPOPT) $< -o $@
	${CPP} -MM ${CPPOPT} -MF $(patsubst %.o,%.d,$@) $<

%.s : %.o
	objdump -d -M,intel  $< > $@

ALL_EXE :=
ALL_EXE += tune1.exe
all: ${ALL_EXE}


OBJLIST :=
OBJLIST += tune1.o

OBJLIST_AVX :=

OBJLIST_FMA :=
OBJLIST_FMA += fma256_noncblas_sgemm_n5.o
OBJLIST_FMA += fma256_noncblas_sgemm_ns5x2.o
OBJLIST_FMA += fma256_noncblas_sgemm_ns2x5.o


-include $(OBJLIST:.o=.d)
-include $(OBJLIST_AVX:.o=.d)
-include $(OBJLIST_FMA:.o=.d)

tune1.exe : $(OBJLIST) $(OBJLIST_AVX) $(OBJLIST_FMA)
	${CPP} $+ -o$@

clean:
	rm *.o *.d $(ALL_EXE)