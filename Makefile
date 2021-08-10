build:
	mpicxx -fopenmp -c -lm finalProject.c -o finalProject.o
	mpicxx -fopenmp -c -lm cFunctions.c -o cFunctions.o
	nvcc -I./inc -c --compiler-options -Wall cudaFunctions.cu -o  cudaFunctions.o 
	mpicxx -fopenmp -o mpiCudaOpemMP  finalProject.o cFunctions.o cudaFunctions.o /usr/local/cuda-9.1/lib64/libcudart_static.a -ldl -lrt

clean:
	rm -f *.o ./mpiCudaOpemMP

run1:
	mpiexec -np 1 ./mpiCudaOpemMP 

run2:
	mpiexec -np 2 ./mpiCudaOpemMP 

runOn2:
	mpiexec -np 2 -machinefile  mf  -map-by  node  ./mpiCudaOpemMP
