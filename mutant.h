#pragma once
#include </usr/local/cuda/include/cuda_runtime.h>
#include </usr/local/cuda/include/device_launch_parameters.h> 

#include "declaration.h"
typedef struct mutant
{
    int offset;
	int position;
	char newLetter;
	double change;
	double totalScore;

}Mutant;

//implementation at cudaFunctions.cu
__device__ __host__ void initMutant (Mutant* m);
__device__ __host__ void copyMutant(Mutant* m1, Mutant* m2);
__device__ __host__  int compareByChange(Mutant* m1, Mutant* m2);
__device__ __host__  int compareByScore(Mutant* m1, Mutant* m2);
__device__ __host__  int isSwapable(int minOrMax,Mutant* m ,Mutant* temp,int(*compare)(Mutant*,Mutant*));

