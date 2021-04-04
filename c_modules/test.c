#include <stdio.h>

#include "utils.h"

#define N 9
#define MAX 6

char test_bools[N][MAX] = {
    "yes", "true", "True", "False", "false", "y", "t", "1", "0"
    };

int main (void) {
    for (size_t i = 0; i < N; i++) {
        int len = strlen(test_bools[i]);
        int chars[len];
        for(int j = 0; j < len; j++) {
            chars[j] = test_bools[i][j];
        }
        if(str2bool(chars, len))
            printf("True\n");
        else
            printf("False\n");
    }
    return 0;
}