#include "utils.h"

#define N 5
#define MAX 5

char trues[N][MAX] = {"yes", "true", "y", "t", "1"};

/**
 * checks a single trues againt each char. If no match - return.
 * Success then only all match.
*/
bool arr_comp(int8_t i, int8_t lowers[], int8_t size) {    
    for(int8_t j=0; j<size; j++)
        if((int8_t)trues[i][j] != lowers[j])
            return false;
    return true;
}

bool str2bool(int8_t v[], int8_t size) {
    int8_t lowers[size];

    for(int8_t i = 0; i < size; i++ ) {
        if ((v[i] >= 65) && (v[i] <= 90))
            lowers[i] = v[i] + 32;
        else
            lowers[i] = v[i];
    }

    for(int8_t i = 0; i < N; i++)
        if(arr_comp(i, lowers, size)) return true;

    return false;
}

// float *diffSoC(float *chargeData, float *discargeData){
//     int i;
//     size_t n = sizeof(chargeData) / sizeof(float);
//     float *res = (float *) malloc(sizeof(float) * n);
//     for(i = 0; i<n;i++)
//         res[i] = round((chargeData[i] - discargeData[i])*100)/100;
//     return res;
// }
int bytes2int(unsigned char bytes[]) {
    unsigned int reassembled_data = 0;
    reassembled_data |= bytes[0] << 24;
    reassembled_data |= bytes[1] << 16;
    reassembled_data |= bytes[2] <<  8;
    reassembled_data |= bytes[3] <<  0; // Converting to volts
    return reassembled_data;
}