// If utils.h is not included to any file
#ifndef UTILS_H

// Declare macro as a flag that specifies utils.h is included
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

// Declare all functions
/**
 * Checks if input string belong to a boolean
*/
bool str2bool(int8_t v[], int8_t size);

#endif // UTILS_H