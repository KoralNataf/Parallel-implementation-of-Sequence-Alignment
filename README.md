# Parallel-implementation-of-Sequence-Alignment
This project is part of Parallel Computation course.

Intro
In bioinformatics, a sequence alignment is a way of arranging the sequences of DNA, RNA, or protein to identify regions of similarity that may be a consequence of functional, structural, or evolutionary relationships between the sequences. [1]

Sequence Alignment Evaluation
Each pair of characters generates a special character that indicates the degree of similarity between them.
The special characters are * (asterisk), : (colon), . (dot), and _ (space).
The following definitions apply:

Equal characters will produce a *.
If two characters are not equal, but present in the same conservative group, they will produce a * sign.
If characters of a pair are not in the same conservative group but are in a semi-conservative group, then they will produce a ..
If none of the above is true, the characters will result in a ' '(space) sign.
Equation
Since each sign is weighted, the following equation will result from comparing two sequences:
S=N1×W1−N2×W2−N3×W3−N4×W4
 
Ni  represents the amount, and  Wi  represents the weight, respectively, of *, :, ., and ' '(space).

Groups
Conservative Groups	
NDEQ	NEQK	STA
MILV	QHRK	NHQK
FYW	HY	MILF

Semi-Conservative Groups
SAG	ATV	CSA
MSGNDILV	STPA	STNK
NEQHRK	NDEQHK	SNDEQK
HFY	FVLIM	

An example of a pair-wise evaluation

PSEKHLQCLLQRHKGK
HSKSHLQHLLQRHKSQ
 *:.*** ******.:
The following can be seen above:

The 2nd pair consists of the characters S and S, they are equal, and hence result in the * sign.
The 3rd pair, E and K, are not equal, but present in the conservative group NEQK, so the result is a :.
The 4th pair, K and S, don't belong to the same conservative group, but rather the same semi-conservative group STNK. Therefore, they result in a . sign.
The 1st pair consists of P and H without applying any of the rules defined above, so they result in the _ sign.
The similarity of two sequences Seq1 and Seq2 defined as followed:

Seq2 is places under the Sequence Seq1 with offset n from the start of Seq1. Where Seq2 do not allowed to pass behind the end of Seq1.
The letters from Seq1 that do not have a corresponding letter from Seq2 are ignored.
The Alignment Score is calculated according the pair-wise procedure described above.
Examples:

LQRHKRTHTGEKPYEPSHLQYHERTHTGEKPYECHQCHQAFKKCSLLQRHKRTH
                  HERTHTGEKPYECHQCRTAFKKCSLLQRHK
                  ****************: ************
Weights: 1.5 2.6 0.3 0.2
Offset: 21
Score: 39.2

ELMVRTNMYTONEWVFNVJERVMKLWEMVKL
MSKDVMSDLKWEV
: .:: :  :* .
Weights: 5 4 3 2
Offset: 3
Score: -31

Mutation
For a given Sequence S we define a Mutant Sequence  MS(n)  which is received by substitution of one or more characters by other character defined by Substitution Rules:

The original character is allowed to be substituted by another character if there is no conservative group that contains both characters.
For example:
N is not allowed to be substituted by H because both characterss present in conservative group NHQK.
N may be substituted by W because there is now conservative group that contains both N and W.
It is not mandatory to substitute all instances of some characters by same substitution character, for example the sequence PSHLSPSQ has Mutant Sequence PFHLSPLQ.

Project Definition
In the given assignment, two sequences Seq1, Seq2, and a set of weights is provided.
A mutation of the sequences Seq2 and it's offset is need to be found, which produce the MAX or MIN score (will be given as an input as well).


Parallel solution: 
We will divide the work between two computers using MPI, and each computer will divide the work between CUDA and OMP.
In this approach each computer will receive half of the offsets and will find the mutant with the best alignment score.

In my solution I chose to calculate in advance and save in matrix the signs of all possible combinations.
The reason is that for very long strings, a huge number of calculations can be reached, for example 5000 letters and 5000 offsets = 25,000,000 calculations.

OMP implementation : 
Each process calculates its own matrix of signs at the beginning of the work. Because the dependence of the calculations is not dependent on any other calculations, OMP is used to calculate the matrix.
In addition, OMP makes full use of its power by parallelizing only the offsets. Therefore, the threads divide the work between them and each one checks for a certain amount of offsets what is the best mutatnt. 
The selection of the best of all the threads is made using a variable update with the help of #pragma omp critical.
When OMP and CUDA are running together, then OMP uses one thread to run CUDA and the three remaining threads divide the work on the offsets. When CUDA is not enabled, all four threads will share the work.

CUDA implementation:
CUDA has a huge number of threads (1024 per block and a maximum of 65,535 blocks).
Because the project is limited in size -
Seq1 - maximum 10000 characters
Seq2 - maximum 5000 characters
the maximum amount of threads we will need to parallel both offsets and letters does not exceed the limits of CUDA.
For paralize all the offsets and all the letters:
each block will serve as an offset, and each thread will calculate the relevant pair in the offset. Since block size is limited to 1024 there may be threads that perform more than one calculation but a maximum of 5 calculations will be performed since Seq2 length is a maximum of 5000 characters.
After that, we will run a reduction twice
Once to find the best mutant for a specific offset and calculate aligenment score.
Second time to find the best mutant for all the offsets based on alignment score.

Parallel Reduction
Parallel reduction refers to algorithms that combine an array of elements to produce a single value.
Among the problems that can be solved by this algorithm are those involving operators that are associative and commutative.
The following are some examples:
-Sum of an array.
-Minimum/Maximum of an array.
If one has an array of  n  values and  n  threads, the reduction algorithm provides a solution of  log(n) .
Reduce an array with  n  elements requires the algorithm to calculate the ceiling number of  n , which is a power of 2 ( m=2⌈log(n)⌉ ).
At the beginning of the algorithm, a  m/2  stride constant is defined.
For each iteration of the algorithm, every cell performs the reduced operation between itself ( i ) and  i+stide . After each iteration, divide stride by 2.

How to run:
The project was developed using MPI, OpenMP, and CUDA. Therefore, all of those library had to be installed for the project to run.
An input file with a name of input.txt, and with the following structure has to be present in the root directory:
The first line will contain 4 weights (decimal or non decimal) in the exact order of W1, W2, W3, and W4.
The seconde line will contains the first sequence Seq1 (up to 10,000 characters).
The third line will contains the second sequence Seq2 (up to 5,000characters).
The last line will contain the string maximum or minimum to define the algorithm which defines the goal of the search.
The output file will results with the mutant of Seq2 in the first line, and it's offset and score in the second line.

To run on 2 computers you must attach a file named mf including ip of computers. The first line is the IP of the computer running the project.
Second row The IP of the second computer.
The project should be compiled on both computers

The Makefile is attached and the following commands can be used:
make build - compile the project.
make run1 - serial running.
make run2 - running on one computer with two processes (50% OMP 50% CUDA)
make runOn2 - running on 2 computers (50% OMP 50% CUDA)

More options:
mpiexec -np 2 ./mpiCudaOpemMP percent -running on one computer with two process(percent is the percent we want OMP made(integer number 0-100)).
mpiexec -np 2 -machinefile  mf  -map-by  node  ./mpiCudaOpemMP percent - running on two computers(percent is the percent we want OMP made(integer number 0-100)).
