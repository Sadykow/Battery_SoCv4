// If utils.h is not included to any file
#ifndef UTILS_H

// Declare macro as a flag that specifies utils.h is included
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

// Declare all functions
/**
 * Checks if input string belong to a boolean
*/
bool str2bool(int8_t v[], int8_t size);

/**
 * Testing conversion..
*/
int bytes2int(unsigned char bytes[]);
// float *diffSoC(float *chargeData, float *discargeData);

#endif // UTILS_H