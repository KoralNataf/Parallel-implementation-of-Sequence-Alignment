#include "myProto.h"
#include <mpi.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

char allSigns[N][N];

/*This function get Data type for struct Data and initialize it.*/
void init_data_type(MPI_Datatype* mpi_data_type)
{
	int blockLength[ARGS_IN_DATA]={1,1,1,WEIGHTS,MAX_SEQ1,MAX_SEQ2};/*number of blocks for each parameter*/
	MPI_Aint displacements[ARGS_IN_DATA]={offsetof(struct data,minOrMax),
										  offsetof(struct data, allOffsets),
										  offsetof(struct data, offsetsForProcess),
										  offsetof(struct data, weights),
										  offsetof(struct data, seq1),
										  offsetof(struct data, seq2)};

	MPI_Datatype types[ARGS_IN_DATA]={MPI_INT,MPI_INT,MPI_INT,MPI_DOUBLE,MPI_CHAR,MPI_CHAR}; 

	MPI_Type_create_struct(ARGS_IN_DATA, blockLength, displacements, types,
			mpi_data_type);
	MPI_Type_commit(mpi_data_type);
}

/*This function get Data type for struct Mutant and initialize it.*/
void init_mutant_type(MPI_Datatype* mpi_mutamt_type)
{
	int blockLength[ARGS_IN_MUTANT]={1,1,1,1,1};/*number of blocks for each parameter*/
	MPI_Aint displacements[ARGS_IN_MUTANT]={offsetof(struct mutant,offset),
										  offsetof(struct mutant, position),
										  offsetof(struct mutant, newLetter),
										  offsetof(struct mutant, change),
										  offsetof(struct mutant, totalScore)};

	MPI_Datatype types[ARGS_IN_MUTANT]={MPI_INT,MPI_INT,MPI_CHAR,MPI_DOUBLE,MPI_DOUBLE}; 
	MPI_Type_create_struct(ARGS_IN_MUTANT, blockLength, displacements, types,
			mpi_mutamt_type);
	MPI_Type_commit(mpi_mutamt_type);
}

/*This function calculate all signs for omp part using omp, it saved in matrix call allSign.
 This matrix is global and in omp part it calculate by each procces only one time when program start.*/
void calcAllSigns()
{
	char letters[] = { 'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','-' };
#pragma omp parallel for
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			allSigns[i][j] = calcSign(letters[i], letters[j]);
}

/*This function print str and color the letter of mutant in red.
 This function not used in this program, it used for debugging while program written.*/
void printStr(char* str,int length,int offset,int mutant)
{
	if(offset < 0 )
		offset=0;
	for(int i=0 ; i<offset; i++)
		printf(" ");//print for offset
	
	for(int i =0 ; i<length; i++)
		if(i == mutant)
			printf("\x1B[31m""%c",str[i]);//color the letter for mutant in red
		else
			printf("\x1B[0m""%c",str[i]);//print other letters in bleck
	printf("\n");
}

/*This function read data from file and initialize struct Data with details from file.*/
void readFromFile(Data* data)
{
	FILE* f = fopen(FILE_NAME,"r");
	if(f == NULL)
	{
		printf("Failed opening the file..\n");
		MPI_Abort(MPI_COMM_WORLD,ERROR_FILE);
	}

	if(fscanf(f,"%lf %lf %lf %lf",&data->weights[W_STAR],&data->weights[W_COLON],&data->weights[W_DOT],&data->weights[W_SPACE]) != WEIGHTS)
	{
		printf("Failed reading weights..\n");
		fclose(f);
		MPI_Abort(MPI_COMM_WORLD,ERROR_READ);
	}

	if(fscanf(f,"%s",data->seq1) != 1)
	{
		printf("Failed reading seq1..\n");
		fclose(f);
		MPI_Abort(MPI_COMM_WORLD,ERROR_READ);
	}

	if (fscanf(f,"%s", data->seq2) != 1)
	{
		printf("Failed reading seq2..\n");
		fclose(f);
		MPI_Abort(MPI_COMM_WORLD, ERROR_READ);
	}

	char goalOfSearch[STR_SIZE];
	if (fscanf(f,"%s", goalOfSearch) != 1)
	{
		printf("Failed reading maximum or minimum..\n");
		fclose(f);
		MPI_Abort(MPI_COMM_WORLD, ERROR_READ);
	}
	
	if(strcmp(goalOfSearch,MINIMUM_STR) == 0)
		data->minOrMax = MINIMUM;
	else if(strcmp(goalOfSearch,MAXIMUM_STR) == 0)
		data->minOrMax = MAXIMUM;
	else
	{
		printf("Uncorrect data in file..\n");
		fclose(f);
		MPI_Abort(MPI_COMM_WORLD,ERROR_READ);
	}
	
	fclose(f);//close input.dat
}

/*This function write best mutant to output file.*/
void writeResultToFile(Mutant* m , Data* data)
{
	FILE *outputFile = fopen(OUTPUT_FILE_NAME, "w");
	if (outputFile == NULL)
	{
		printf("Failed opening the output file..\n");
		MPI_Abort(MPI_COMM_WORLD,ERROR_FILE);
	}
	if(m->newLetter == CHAR_NOT_FOUND)
		fprintf(outputFile, "There is no option to create mutant.\nBest offset: %d Score: %g",m->offset,m->totalScore);
	else
		{
			data->seq2[m->position]=m->newLetter;
			fprintf(outputFile, "Mutant: %s\nOffset: %d	Score: %g",data->seq2,m->offset,m->totalScore);
		}
	fclose(outputFile);
}

/*This function get lenth of 2 strings and return number of offsets between string1 to string2.
 Offset is the position that string2 is placed relative string1.*/
int culcNumOfOffsets(int seq1_len,int seq2_len)
{
	return seq1_len - seq2_len + 1;
}

/*This function check if number of offset os even , if it odd it return half+1 - it the part from offset
 that other procces/omp/cuda should work on.*/
int getSecondHalf(int numberOfOffsets)
{
	if(numberOfOffsets % 2 == 0)
		return numberOfOffsets/2;
	else
		return numberOfOffsets/2 +1;
}

/*This function running about all pares in specific offset and choose the best change
 of letter,it minning the best mutant for this offset( in omp part ). 
 The choosing of mutant it by the relative change that effect on total score of this mutant.
 While running it calculate total score of this mutant.*/
void findBestForOffsetOmp(Mutant* m,int offset,Data* data)
{
	double score = 0;// start with score 0
	Mutant temp;
	initMutant(&temp);//create temp mutant

	temp.offset = offset;//update offset field
	int size = strlen(data->seq2);
	for (int i = 0; i < size  ; i++)
	{
		temp.position = i;//tpdate position field
		char sign = allSigns[convertChartoInt(data->seq1[offset + i])][convertChartoInt(data->seq2[i])];//get sign of 2 letters
		if(data->minOrMax == MINIMUM)
			swapMinCharacter(&temp,data->seq1[offset + i],data->seq2[i],sign,data->weights);//get the best change for this 2 letters
		
		else 
			swapMaxCharacter(&temp, data->seq1[offset + i], data->seq2[i], sign,data->weights);//get the best change for this 2 letters
		score += temp.totalScore;//while running on seq2 and looking for best mutant we calculate total score.
	
		if(isSwapable(data->minOrMax,m,&temp,compareByChange))//check if we find a better mutant
			copyMutant(m,&temp);//copy mutant
	}
	m->totalScore =score + m->change; // upadate total score after found best change
}

/*This function running about all offsets and choose the best mutant by total score. 
 It using pragma omp parallel for and critical to update the best mutant.*/
void getBestMutantOmp(Mutant* m,Data* data,int startOffset,int endOffset)
{	
	Mutant temp;
		for(int curOffset = startOffset; curOffset< endOffset; curOffset++)
		{
			initMutant(&temp); // Data Reset for temp
			findBestForOffsetOmp(&temp, curOffset ,data);
	#pragma omp critical
	{// m is common for all threads - so access m need to be serial.
		if (isSwapable(data->minOrMax,m,&temp,compareByScore))//chaeck if we found better mutant
			copyMutant(m, &temp);//copy the best mutant		
	}
		}
}

/*This function is managed each procces job, calculate omp and cuda part and execute 
 each part.*/	
void startProcessJob(Mutant* m ,Data* data , int rank,int size,int precentOmp)
{
	Mutant ompRes,cudaRes;//mutant for each part omp and cuda
	initMutant(&ompRes);
	initMutant(&cudaRes);
	//calc start and end for each process
	int startOffsetForProcess = rank * (data->allOffsets/size);
	int endOffsetForProcess = startOffsetForProcess + data->offsetsForProcess;
	//calc start and end for omp part
	int ompPart = data->offsetsForProcess*precentOmp/100; 
	int ompStartOffset = startOffsetForProcess;
	int ompEndOffset = ompStartOffset + ompPart;
	//calc start and part for cuda 
	int cudaStartOffset = ompEndOffset;
	int cudaPart = data->offsetsForProcess - ompPart;
#pragma omp parallel
{
	int tId = omp_get_thread_num();
	int numOfThreads = omp_get_num_threads();

	if(size > 1 && cudaPart > 0 && tId == 0)
		computeOnGPU(&cudaRes,data,cudaStartOffset,cudaPart);
	
	if(ompPart > 0)
	{
		if(cudaPart > 0)
		{
			numOfThreads--;
			tId--;
		}

		if(tId != -1)
		{
			int offsetsForThread = ompPart / numOfThreads;
			int startThread =startOffsetForProcess+ tId * offsetsForThread;
			if(ompPart % numOfThreads != 0 && tId == numOfThreads-1)
				offsetsForThread +=ompPart % numOfThreads;
			int endThread = startThread + offsetsForThread;
			getBestMutantOmp(&ompRes,data,startThread,endThread);
		}
	}	
}
if(isSwapable(data->minOrMax,&ompRes,&cudaRes,compareByScore))
	copyMutant(m, &cudaRes);
else
	copyMutant(m, &ompRes);
	
}