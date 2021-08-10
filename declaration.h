#pragma once

#define FILE_NAME "input.txt"
#define OUTPUT_FILE_NAME "output.txt"
#define MASTER 0
#define MAX_SEQ1 10001
#define MAX_SEQ2 5001
#define WEIGHTS 4
#define STR_SIZE 10
#define ASCII 65

#define STAR '*'
#define COLON ':'
#define DOT '.'
#define SPACE ' '
#define HYPHEN '-'
#define H_INDEX 26

#define W_STAR 0
#define W_COLON 1
#define W_DOT 2
#define W_SPACE 3

#define BUFFER_SIZE 15100
#define TAG_SEND 0
#define CONSERVATIVE 9
#define SEMI_CONSERVATIVE 11
#define N 27 //number of letters in english + hyphen

#define MINIMUM 0
#define MAXIMUM 1

#define MINIMUM_STR "minimum"
#define MAXIMUM_STR "maximum"

#define TAG 0
#define ARGS_IN_DATA 6
#define ARGS_IN_MUTANT 5
#define CHAR_NOT_FOUND '\0'

#define MAX_BLOCK 1024
#define N_POW_2 32

typedef enum {ERROR_ARGS=1,ERROR_FILE,ERROR_READ} errorsTypes;