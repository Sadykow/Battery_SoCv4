#include <stdio.h>

//>> Read CSV
#include <stdlib.h>
#include <string.h>
// <<
#include "utils.h"

#define N 9
#define MAX 6
#define MAXCHAR 1000

char test_bools[N][MAX] = {
        "yes", "true", "True", "False", "false", "y", "t", "1", "0"
    };
//>> Read CSV
const char* getfield(char* line, int num) {
    const char* tok;
    for (tok = strtok(line, ";");
    tok && *tok;
    tok = strtok(NULL, ";\n")) {
        if (!--num)
            return tok;
    }
    return NULL;
}
// <<
int main (void) {
    // for (size_t i = 0; i < N; i++) {
    //     int len = strlen(test_bools[i]);
    //     int chars[len];
    //     for(int j = 0; j < len; j++) {
    //         chars[j] = test_bools[i][j];
    //     }
    //     if(str2bool(chars, len))
    //         printf("True\n");
    //     else
    //         printf("False\n");
    // }
    // float arr1[10] = {1,2,3,4,5,6,7,8,9,10};
    // float arr2[10] = {10,9,8,7,6,5,4,3,2,1};

    // float *arr3 = diffSoC(arr1, arr2);
    // for(int i = 0; i < 10; i++) {
    //     printf("%f ", arr3[i]);
    // }
    // printf("\n");
    // FILE* stream = fopen("../Models/Chemali2017/FUDS-models/history-FUDS2.csv", "r");
    // char line[MAXCHAR];
    // char *token;
    // int index = 0;
    // // Write the finder of the minimum epoch.
    // while (feof(stream) != true)
    // {
    //     fgets(line, MAXCHAR, stream);
    //     if(index == 0) {
    //         printf("Row: %s", line);
    //     } else {
    //         token = strtok(line, ",");
    //         while(token != NULL) {
    //             //printf("%s ", token);
    //             printf("fl: %4.8f " ,atof(token)); 
    //             token = strtok(NULL, ",");
    //         }
    //     }
    //     printf("\n");
    //     index++;
    // }
    // Example raw data from SFP module
    // unsigned char byte1 = 0xFC;
    // unsigned char byte2 = 0x0C;
    // unsigned char byte3 = 0x7F;
    // unsigned char byte4 = 0xC9; // Assembling to 32 bit unsigned integer
    // unsigned int reassembled_data = 0;
    // reassembled_data |= byte1 << 24;
    // reassembled_data |= byte2 << 16;
    // reassembled_data |= byte3 <<  8;
    // reassembled_data |= byte4 <<  0; // Converting to volts
    // float voltage = (int)(reassembled_data) / 1000000.0f; // Calculated value is -12.213964 Volts
    // printf("%i :: %f\n", (int)(reassembled_data),voltage);
    unsigned char bytes[] = {0xFC, 0x0C, 0x7F, 0xC9};
    unsigned int data = bytes2int(bytes);
    float voltage = (int) data / 1000000.0f;
    printf("%u %i:: %f\n", data, (int)data, voltage);
    return 0;
}