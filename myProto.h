#pragma once
#include "declaration.h"
#include "mutant.h"
#include "data.h"
#include "cudaProto.h"
#include <mpi.h>

void init_data_type(MPI_Datatype* mpi_data_type);
void init_mutant_type(MPI_Datatype* mpi_mutamt_type);
void calcAllSigns();
void printStr(char* str,int length,int offset,int mutant);
void readFromFile(Data* data);
void writeResultToFile(Mutant* m , Data* data);
int culcNumOfOffsets(int seq1_len,int seq2_len);
int getSecondHalf(int numberOfOffsets);
void findBestForOffsetOmp(Mutant* m,int offset,Data* data);
void getBestMutantOmp(Mutant* m,Data* data,int startOffset,int endOffset);
void startProcessJob(Mutant* m ,Data* data , int rank,int size, int precentOmp);