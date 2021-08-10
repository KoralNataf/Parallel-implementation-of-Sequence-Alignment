/*
 ============================================================================
 Name        : FinalProject.c
 Author      : Koral Nataf 208726257
 ============================================================================
 */
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#include <stdlib.h>
#include "declaration.h"
#include "myProto.h"
#include "mutant.h"

int main(int argc, char *argv[]) {
    int size, rank;
    Data data;
    Mutant m;
    MPI_Status  status;
    int precentOmp = 50;

    omp_set_num_threads(4);//set num of threads to omp

    if(argc > 2)
    {
      printf("number of arguments cannot be grater then 2 ..\n");
      MPI_Abort(MPI_COMM_WORLD,ERROR_ARGS);
    }else if(argc == 2)
        precentOmp= atoi(argv[1]);

    /*data type definitions*/
	MPI_Datatype mpi_data_type;
    MPI_Datatype mpi_mutant_type;
 
    calcAllSigns();
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size > 2 || size < 1) {
       printf("Execute code with maximum processes\n");
       MPI_Abort(MPI_COMM_WORLD, __LINE__);
    }else if(size == 1)
    {
        omp_set_num_threads(1);//serial running
        precentOmp =100; // serial running
    }
  

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //initalize all Data type
    init_data_type(&mpi_data_type);
    init_mutant_type(&mpi_mutant_type);

    //Divide the tasks between both processes
    if (rank == MASTER)
    {
    	readFromFile(&data);// only master reading from file
        data.allOffsets=culcNumOfOffsets(strlen(data.seq1),strlen(data.seq2));
        MPI_Bcast(&data,1,mpi_data_type,MASTER,MPI_COMM_WORLD);
        data.offsetsForProcess = data.allOffsets/size;
        
    }
    else {
        MPI_Bcast(&data,1,mpi_data_type,MASTER,MPI_COMM_WORLD);
        data.offsetsForProcess = getSecondHalf(data.allOffsets);
    }

    initMutant(&m);
    startProcessJob(&m,&data,rank,size,precentOmp);

    if(rank == MASTER)
    {
        if(size > 1)//if number of procces is 2 then MASTER need get best option of other procces
        {
            Mutant other;
            MPI_Recv(&other,1,mpi_mutant_type, 1 ,TAG,MPI_COMM_WORLD,&status);
        	if (isSwapable(data.minOrMax,&m,&other,compareByScore))
			    copyMutant(&m, &other);
        }
        writeResultToFile(&m,&data);
    }
    else
         MPI_Send(&m , 1 ,mpi_mutant_type ,MASTER ,TAG ,MPI_COMM_WORLD);
    
    //free all Data type
    MPI_Type_free(&mpi_data_type);
    MPI_Type_free(&mpi_mutant_type);
    
    MPI_Finalize();

    return 0;
}
