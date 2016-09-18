CC = gcc
CPP = g++
CU = nvcc 
CUDADIR = /usr/local/cuda

VERS = _afh_1

#LIBS = -lm
INC1 = -I${CUDADIR}/include
INCDIRS = ${INC1}

LFLAGS = -L${CUDADIR}/lib64 -lcuda -lcudart -lcublas
CFLAGS = ${INCDIRS}
#CUFLAGS = --ptxas-options=-v -gencode arch=compute_20,code=sm_20
#CUFLAGS = --ptxas-options=-v

##########
.SUFFIXES: .cu .cpp .c .o .exe
.PHONY: clean cleanx

#%.o: %.cu
#	$(CU) $(CFLAGS) -c $<
#%.s: %.cu
#	$(CU) $(CFLAGS) -S $< 
#%.o: %.cpp
#	$(CPP) $(CFLAGS) -c $<
#%.s: %.cpp
#	$(CPP) $(CFLAGS) -S $< 
#%.o: %.c
#	$(CC) $(CFLAGS) -c $<
#%.s: %.c
#	$(CC) $(CFLAGS) -S $< 
#%.exe: %.o
#	$(CC) -o $@ $(LFLAGS) $^ 

##########
scn: scn$(VERS).exe
	echo 'scn$(VERS) done'
scn$(VERS).exe: scn.o scn_aux.o kim_aux.o kim_kernels.o df_aux.o df_kernels.o utils.o
	$(CPP) -o $@ $(LFLAGS) $^
scn.o: scn.cpp scn.h kim.h df.h kim_aux.h df_aux.h kim_kernels.h df_kernels.h parameters.h
	$(CPP) $(LFLAGS) $(CFLAGS) -c $<
scn_aux.o: scn_aux.cpp scn.h kim.h utils.h
	$(CPP) $(CFLAGS) -c $<
kim_aux.o: kim_aux.cpp kim.h scn.h utils.h parameters.h
	$(CPP) $(CFLAGS) -c $<
kim_kernels.o: kim_kernels.cu kim.h scn.h parameters.h
	$(CU) $(CUFLAGS) $(CFLAGS) -c $<
df_aux.o: df_aux.cpp df.h scn.h utils.h parameters.h kim.h
	$(CPP) $(CFLAGS) -c $<
df_kernels.o: df_kernels.cu df.h scn.h parameters.h
	$(CU) $(CUFLAGS) $(CFLAGS) -c $<
utils.o: utils.cpp
	$(CPP) $(CFLAGS) -c $<

##########
clean:
	rm *.exe; rm *.o;
cleanx:
	rm pbs*.*; rm out*.txt;
