#include "cudaProto.h"
#include <helper_cuda.h>

__device__ char cudaSigns[N][N];

//Mutant functions

/*This funcion set default values for mutant.*/
__device__ __host__ void initMutant (Mutant* m)
{
    m->offset = -1;
	m->position = -1;
	m->change = 0;
	m->totalScore = 0;
	m->newLetter=CHAR_NOT_FOUND;
}

/*This function copy m2 to m1.*/
__device__ __host__ void copyMutant(Mutant* m1, Mutant* m2)
{
    m1->offset = m2->offset;
	m1->position = m2->position;
    m1->newLetter = m2->newLetter;
	m1->change = m2->change;
	m1->totalScore = m2->totalScore;
}

/*This function compare bwtween 2 mutants by the relative change:
 if m1 > m2 ---> -1
 if equals  --->  0
 if m1 < m1 ---> 1
 */
__device__ __host__  int compareByChange(Mutant* m1, Mutant* m2)
{
	if (m1->change == m2->change)
		return 0;
	else if (m1->change > m2->change)
		return 1;
	else
		return -1;
}

/*This function compare bwtween 2 mutants by the total score:
 if m1 > m2 ---> -1
 if equals  --->  0
 if m1 < m1 ---> 1
 */
__device__ __host__  int compareByScore(Mutant* m1, Mutant* m2)
{
	if (m1->totalScore == m2->totalScore)
		return 0;
	else if (m1->totalScore > m2->totalScore)
		return 1;
	else
		return -1;
}

/*This function check if  temp is better than origin mutant , it check if the it has a mutant and compare
 by compare function that send - it can be by score or by relative change.*/
__device__ __host__  int isSwapable(int minOrMax,Mutant* m ,Mutant* temp,int(*compare)(Mutant*,Mutant*))
{
	if(temp->newLetter == CHAR_NOT_FOUND)// temp has no mutant
		return 0;
	else if(m->newLetter==CHAR_NOT_FOUND)// m has no mutant but temp has
		return 1;
	else if (minOrMax== MAXIMUM && compare(m, temp) < 0 
	|| minOrMax==MINIMUM && compare(m, temp) > 0)//m and temp has mutant - we want the better for us
		return 1;
	else
		return 0;
}
//end of Mutant functions
/////////////////////////////////////////////////////////////////////////////////

//function for device and host
/*For cuda using has no string function, so I implements strlen.
 It get string and return his length.*/
__device__ __host__ int myStrlen(const char *str)
{
    int len;
    for (len= 0; str[len]; len++);
    return len;
}

/*For cuda using has no string function, so I implements strchr.
 It get string and char, and return if this string include this char.*/
__device__ __host__ int myStrchr(const char *str,char c)
{
    int len=myStrlen(str);
    for (int i=0; i<len; i++)
        if(str[i]==c)
            return 1;
    return 0;
}

/*This function checl if two chars is consevative.*/
__device__ __host__ int isConsevative(char s1, char s2)
{
	char conservative[][STR_SIZE] = { "NDEQ","MILV","FYW","NEQK","QHRK","HY","STA","NHQK","MILF" };
	
	for (int i = 0; i < CONSERVATIVE; i++)
	{
		if (myStrchr(conservative[i], s1) && myStrchr(conservative[i], s2))
			return 1;
	}

	return 0;
}

/*This function checl if two chars is semi-consevative.*/
__device__ __host__ int isSemiConsevative(char s1, char s2)
{
	char semiConservative[][STR_SIZE] = { "SAG","SGND","NEQHRK","ATV","STPA","NDEQHK","HFY","CSA","STNK","SNDEQK","FVLIM" };

	for (int i = 0; i < SEMI_CONSERVATIVE ; i++)
	{
		if (myStrchr(semiConservative[i], s1) && myStrchr(semiConservative[i], s2))
			return 1;
	}
	return 0;
}

/*This function get two chars and calc their sign.
 equals ---> '*'
 in same consevative group ---> ':'
 in same semi-consevative  ---> '.'
 else ---> ' '
 */
__device__ __host__ char calcSign(char x, char y)
{
	if (x == y)
		return STAR;
	
	if( x== HYPHEN || y == HYPHEN)// hypen with any char except hiself is space
		return SPACE; 

	if(isConsevative(x,y))
		return COLON;

	if(isSemiConsevative(x,y))
		return DOT;

	return SPACE;
}

/*This function convert char to his position in matrix
(row or cal order by ABC and hypen in end)*/
__device__ __host__  int convertChartoInt(char x)
{
	if(x == HYPHEN)
		return H_INDEX;
	return x - ASCII;
}

/*This function convert position in matrix to correct char 
(row or cal order by ABC and hypen in end)*/
__device__ __host__  char convertIntToChar(int x)
{
	if( x < 0 || x > H_INDEX)
		x=0;

	if(x == H_INDEX)
		return HYPHEN;
	return (char)(x + ASCII);
}

/*This function check who call her - GPU or CPU and know to
 get the correct matrix of sign (in CPU or GPU) , return the sign in matrix 
 that in pisition [index1][index2] */
__device__ __host__ char getSign(int index1,int index2)
{
    #if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ > 0))
        return cudaSigns[index1][index2];
    #else
        return allSigns[index1][index2];
    #endif
}

/*This function get sign and retun his weight*/
__device__ __host__ double getScore(char sign,double* weights)
{
	if (sign == STAR)
		return weights[W_STAR];
	if (sign == COLON)
		return (-weights[W_COLON]);
	if (sign == DOT)
		return (-weights[W_DOT]);
	if (sign == SPACE)
		return (-weights[W_SPACE]);
    return 0;
}

/*This function get:
 s1 s2 -> chars from seq1 and seq2 respectively
 sign -> the sign that we want to get from calc s1 and s2
 and check if we can swap s2 with another char(from the chars that in matrix) - 
 that the result of clculate s1 and new char getting this sign*/
__device__ __host__ char getChangeToSign(char s1, char s2,char sign)
{
	int indexS1 = convertChartoInt(s1);//get the index of the letter in matrix
	int indexS2 = convertChartoInt(s2);//get the index of the letter in matrix

	for (int i = 0; i < N; i++)
	{
		char c = getSign(indexS1,i);
		if (c == sign)
		{
			if (getSign(s2,i) != COLON)// if result id colon it cannot change
				 return convertIntToChar(i);
		}
	}
	return CHAR_NOT_FOUND;//if we cange change return CHAR_NOT_FOUND
}

/*This function get:
 m -> mutant to initialize after change
 s1 s2 -> chars from seq1 and seq2 respectively
 sign1 sign2 -> the sign that we try to get from calculate s1 and s2 , we prefer sign1 on sign2
 change 1 change 2 -> the relative change if we change to sign1 or sign2 respectively
 It try to change to sign1 , if it impossible is try change to sign2, if both Impossible it return CHAR_NOT_FOUD.*/
__device__ __host__ void getBestChange(Mutant* m, char s1, char s2, char sign1, char sign2,double change1 , double change2)
{	
	char letter = getChangeToSign(s1, s2, sign1); // try find a letter that give us first change (better)

	if (letter == CHAR_NOT_FOUND)
	{
		letter = getChangeToSign(s1, s2, sign2);// if failed try to fing a letter that give us other change
		if (letter == CHAR_NOT_FOUND)
		{
			m->newLetter = CHAR_NOT_FOUND;
			return;
		}
			
		m->newLetter = letter;
		m->change = change2;
	}
	else
	{
		m->newLetter = letter;
		m->change = change1;
	}
}

/*This function get:
 s1 s2 -> chars from seq1 and seq2 respectively
 sign -> the result of calculate s1 ,s2 sign
 weights -> array of weights for all signs
 This function return the best swap of s2 it can change for MAXIMUM score
*/
__device__ __host__ void swapMaxCharacter(Mutant* m,char s1, char s2,char sign, double* weights)
{	
	//w1->'*' ,w2->':' ,w3->'.' ,w4->' ' 
	double changeToSpace, changeToDot;
    m->totalScore =getScore(sign,weights);
	switch (sign)
	{
	case STAR:
		changeToSpace = (-weights[W_STAR] - weights[W_SPACE]);//w1-w4
		changeToDot = (-weights[W_STAR] - weights[W_DOT]);//w1-w3
		break;
	case COLON:
		changeToSpace = weights[W_COLON] - weights[W_SPACE];//w2-w4
		changeToDot = weights[W_COLON] - weights[W_DOT];//w2-w3
		break;
	case SPACE: //it must change to *
		m->change = weights[W_SPACE] + weights[W_STAR];//w4+w1
		m->newLetter = s1;
		return;
	case DOT: //it must change to *
		m->change= weights[W_DOT] + weights[W_STAR];//w3+w1
		m->newLetter = s1;
		return;
	default:
		return;
	}

	if (changeToSpace < changeToDot)
			getBestChange(m, s1, s2, DOT, SPACE, changeToDot, changeToSpace);
			
		else
			getBestChange(m, s1, s2, SPACE, DOT, changeToSpace, changeToDot);
}

/*This function get:
 s1 s2 -> chars from seq1 and seq2 respectively
 sign -> the result of calculate s1 ,s2 sign
 weights -> array of weights for all signs
 This function return the best swap of s2 it can change for MINIMUM score
*/
__device__ __host__  void swapMinCharacter(Mutant* m, char s1, char s2, char sign, double* weights)
 {//w1 ->'*' ,w2->':' ,w3->'.' ,w4->' ' 
	double change1 , change2, change3;
	char sign1 ,sign2;
    m->totalScore = getScore(sign,weights);
	switch (sign)
	{
	case STAR: //can be changed to space or dot
		change1 = (-weights[W_STAR] - weights[W_SPACE]);//change to space : -w1-w4
		change2=  (-weights[W_STAR] - weights[W_DOT]);//chane to dot : -w1-w3
		sign1 = SPACE;
		sign2 = DOT;
		break;
	case SPACE: // can be changed to colon or dot
		change1 = weights[W_SPACE] - weights[W_COLON];//change to colon: w4-w2
		change2 =  weights[W_SPACE] - weights[W_DOT];//change to dot: w4-w3
		change3 = weights[W_SPACE] + weights[W_STAR];//change to star w4+w1
		sign1 = COLON;
		sign2 = DOT;
		break;
	case DOT: //can be change to colon or space
		change1 =  weights[W_DOT] - weights[W_COLON];//change to colon: w3-w2
		change2 = weights[W_DOT] - weights[W_SPACE];//change to space: w3-w4
		change3 = weights[W_DOT] + weights[W_STAR];//change to star w3+w1
		sign1=COLON;
		sign2=SPACE;
		break;
	case COLON: // can be change to space or dot
		change1 = weights[W_COLON] - weights[W_SPACE];//change to space: w2-w4
		change2 = weights[W_COLON] - weights[W_DOT];//change to dot: w2-w3
		sign1= SPACE;
		sign2=DOT;
		break;
	default:
		break;
	}

	if(change1 > change2)
		getBestChange(m, s1, s2, sign2, sign1, change2, change1);
	else
		getBestChange(m, s1, s2, sign1, sign2, change1, change2);

	if(m->newLetter== CHAR_NOT_FOUND && (sign == DOT || sign == SPACE))
	{//if in space or dot we could'nt swap then we should change to star
		m->change = change3;
		m->newLetter = s1;
	}
}

/*This function get maximum pow of 2 that lower from x but maimum 1024*/
__device__ __host__ int getMaxPow2(int x)
{
	int max=1;
	while(max*2 <= x && max*2 <=MAX_BLOCK)
		max*=2;
	return max;
}
//end of functions for device and host
/////////////////////////////////////////////////////////////////////////////////

//functions that call from CPU and execute on GPU
/*This function calculate all signs for cuda part using cuda threds, it saved in matrix call cudaSign.
 This matrix is global and in cuda part it calculate by each procces only one time when cuda start.*/
__global__ void calcCudaAllSigns()
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row < N && col < N)//if threads row and col not out of boands
    	cudaSigns[row][col]= calcSign(convertIntToChar(row),convertIntToChar(col));
}

/*In this function each thread calc the best swap for pair from seq1 and seq2.
  If seq2 length >1024 (maximum block size) so the threads(or part of them) calc more then one pair.
  If thead calc more then one he sum the score in his place in mutant array.
  */
__global__  void calcCudaPart(Data* d_data,int len_seq2,Mutant* d_allMutants,int startOffset) {
   
	int tId = blockIdx.x * blockDim.x+  threadIdx.x ; //calc thread id
	int block = blockIdx.x ; // block number
    int position = threadIdx.x; // position of thread in his offset in the array
	int eachTreadWork = len_seq2/ blockDim.x ; //number of pairs each thread should culc
	if( len_seq2 % blockDim.x !=0)
		eachTreadWork++; // if number is not pow of 2 - part of threads need calculate one mor pair
	int offset =startOffset+ block;//cuda not calc from offset 0
	
    char s1 ,s2;
	Mutant temp;
	d_allMutants[tId].offset = offset;
	d_allMutants[tId].newLetter=CHAR_NOT_FOUND;
	d_allMutants[tId].totalScore=0;
    for(int i =0 ; i<eachTreadWork ; i++)
    {
		if(position +i*blockDim.x >= len_seq2)
			break;
		s1= d_data->seq1[offset + position +i*blockDim.x];
    	s2 = d_data->seq2[position+ i*blockDim.x];
		temp.newLetter = CHAR_NOT_FOUND;
		temp.position =  position +i*blockDim.x;
        threadTask(&d_allMutants[tId],&temp,s1,s2,d_data->weights,d_data->minOrMax);
    }
}

/*This function made reduction for each offset , each block is one offset and reduction sum his score
 with field totalScore and save the best mutant in the start of each block*/
__global__ void reductionInsideOffset(Mutant* allMutants,int minOrMax)
{
	int tId = blockDim.x * blockIdx.x + threadIdx.x;//thread id
	if(threadIdx.x % 2 != 0)//only even id made task
		return;
	 __syncthreads();
	for (int i = 1; i < blockDim.x; i *= 2)
	{
		if (threadIdx.x % (i * 2) == 0)
		{
			allMutants[tId].totalScore += allMutants[tId + i].totalScore;//sum the score
			if (isSwapable(minOrMax,&allMutants[tId], &allMutants[tId + i],compareByChange))
			{//if othet mutant is better then save his details
				allMutants[tId].position = allMutants[tId + i].position;
				allMutants[tId].newLetter = allMutants[tId + i].newLetter;
				allMutants[tId].change = allMutants[tId + i].change;
			}
		}
		 __syncthreads();//syncronyzed inside block
	}
	allMutants[tId].totalScore += allMutants[tId].change;//adding best change
}

/*This function made reduction for all offset , each block has count of offsets and reduction 
 save the best mutant in the start of each block - in the end we running on all blocks and save 
 the best mutant in allMutant[0]*/
__global__ void reductionAllOffset(Mutant* allMutants,int length,int blockSize,int minOrMax)
{
	int tId = blockDim.x * blockIdx.x + threadIdx.x;//thread id
	int offsetIndex =tId * blockSize;//the position of the offset that each thread start with him
	if(threadIdx.x %2 !=0)
		return;

	 __syncthreads();
	for (int i = 1; i < blockDim.x ; i *= 2)
	{
		int nextOffset = offsetIndex+i*blockSize;//the position of the offset that should compare with
		if (tId % (i * 2) == 0 && nextOffset < length)
			if(isSwapable(minOrMax,&allMutants[offsetIndex],&allMutants[nextOffset],compareByScore))
				copyMutant(&allMutants[offsetIndex],&allMutants[nextOffset]);
	 __syncthreads();
	}

	if( tId == 0)
	{//each block save his best - now we need to run and save the best in allMutant[0]
		for( int i =1 ; i < gridDim.x ; i++)
			if(isSwapable(minOrMax,&allMutants[0],&allMutants[i*blockSize*blockDim.x],compareByScore))
				copyMutant(&allMutants[0],&allMutants[i*blockSize*blockDim.x]);
	}

}
//end of global functions
/////////////////////////////////////////////////////////////////////////////////

//device functions
/*In this function each thead culc the mutant of pair , sum their score and save the position,letter 
 and change.*/
__device__ void threadTask(Mutant* m,Mutant* temp,char s1,char s2,double* weights,int minOrMax)
{
	if(minOrMax == MAXIMUM)
        swapMaxCharacter(temp,s1,s2,getSign(convertChartoInt(s1),convertChartoInt(s2)),weights);
    else
        swapMinCharacter(temp,s1,s2,getSign(convertChartoInt(s1),convertChartoInt(s2)),weights);
	m->totalScore+=temp->totalScore;
	if(isSwapable(minOrMax,m,temp,compareByChange))
	{
		m->position = temp->position;
		m->newLetter = temp->newLetter;
		m->change = temp->change;
	}

}
//end of device functions
/////////////////////////////////////////////////////////////////////////////////

//main funtion of cuda
int computeOnGPU(Mutant* m ,Data* data,int startOffset,int cudaPart) 
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    size_t d_size = sizeof(Data);

    // Allocate memory on GPU to copy the data from the host
    Data *d_data;
    err = cudaMalloc((void **)&d_data,d_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr,"Failed to allocate device memory - %s\n",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Copy data from host to the GPU memory
    err = cudaMemcpy(d_data, data , d_size , cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr,"Failed to copy data from host to device - %s\n",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    } 

    //fill signs matrix
    dim3 threadsPerBlockSigns(N_POW_2,N_POW_2);
    dim3 numBlockSigns(1,1);
    calcCudaAllSigns<<<numBlockSigns,threadsPerBlockSigns>>>();

    //calc size of block and count of blocks for all tasks
	int len_seq2 =myStrlen(data->seq2);
    int threadsPerBlock=getMaxPow2(len_seq2);
    int blocksPerGrid =cudaPart;
	
	//create all mutants array
	d_size = sizeof(Mutant);
    Mutant *d_allMutants;
    err = cudaMalloc((void **)&d_allMutants,blocksPerGrid * threadsPerBlock * d_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr,"Failed to allocate device memory - %s\n",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	//calculate all pairs for all procceses
    calcCudaPart<<<blocksPerGrid, threadsPerBlock>>>(d_data,len_seq2,d_allMutants,startOffset);
	
	//calc best score for each offset and find best mutant for each offset- saved in start of block
	reductionInsideOffset<<<blocksPerGrid ,threadsPerBlock>>>(d_allMutants,data->minOrMax);
	
	//new configure of threads - need threads for offsets count
	int newThreadsPerBlock = getMaxPow2(cudaPart);
    int newBlocksPerGrid =(cudaPart + newThreadsPerBlock - 1) / newThreadsPerBlock;

	//reduction to all offset - best mutant saved in d_allMutants[0]
	reductionAllOffset<<<newBlocksPerGrid, newThreadsPerBlock>>>(d_allMutants,blocksPerGrid*threadsPerBlock,threadsPerBlock,data->minOrMax);
	
	//copy best mutant cuda found to CPU
	err = cudaMemcpy(m, &d_allMutants[0] , d_size , cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr,"Failed to copy data from device to host - %s\n",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    } 

	//free all alocate memory
	if (cudaFree(d_data) != cudaSuccess)
    {
        fprintf(stderr,"Failed to free device d_data - %s\n",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	if (cudaFree(d_allMutants) != cudaSuccess)
    {
        fprintf(stderr,"Failed to free device allMutants - %s\n",cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

	cudaDeviceSynchronize();

    return 0;
}

