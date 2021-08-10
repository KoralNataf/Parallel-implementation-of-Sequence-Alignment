#pragma once
#include </usr/local/cuda/include/cuda_runtime.h>
#include </usr/local/cuda/include/device_launch_parameters.h>
#include "declaration.h"
#include "mutant.h"
#include "data.h"

extern char allSigns[N][N];

//function for device and host
__device__ __host__ int myStrlen(const char *str);
__device__ __host__ int myStrchr(const char *str,char c);
__device__ __host__ int isConsevative(char s1, char s2);
__device__ __host__ int isSemiConsevative(char s1, char s2);
__device__ __host__ char calcSign(char x, char y);
__device__ __host__  int convertChartoInt(char x);
__device__ __host__  char convertIntToChar(int x);
__device__ __host__ char getSign(int index1,int index2);
__device__ __host__ double getScore(char sign,double* weights);
__device__ __host__ char getChangeToSign(char s1, char s2,char sign);
__device__ __host__ void getBestChange(Mutant* m, char s1, char s2, char sign1, char sign2,double change1 , double change2);
__device__ __host__ void swapMaxCharacter(Mutant* m,char s1, char s2,char sign, double* weights);
__device__ __host__  void swapMinCharacter(Mutant* m, char s1, char s2, char sign, double* weights);
__device__ __host__ int getMaxPow2(int x);

//functions that call from CPU and execute on GPU
__global__ void calcCudaAllSigns();
__global__  void calcCudaPart(Data* d_data,int len_seq2,Mutant* d_allMutants,int startOffset);
__global__ void reductionInsideOffset(Mutant* allMutants,int minOrMax);
__global__ void reductionAllOffset(Mutant* allMutants,int length,int blockSize,int minOrMax);

//device functions
__device__ void threadTask(Mutant* m,Mutant* temp,char s1,char s2,double* weights,int minOrMax);

//main funtion of cuda
int computeOnGPU(Mutant* m ,Data* data,int startOffset,int cudaPart);

