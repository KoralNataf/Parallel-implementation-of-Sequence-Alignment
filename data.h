#pragma once 
#include "declaration.h"

typedef struct data
{
    int minOrMax;
    int allOffsets;
    int offsetsForProcess;
    double weights[WEIGHTS];
    char seq1[MAX_SEQ1];
    char seq2[MAX_SEQ2];
    
}Data;



