#include <stdio.h>

void dot_product(const float* a, const float* b, float* c, int d) {
    for (int e = 0; e < d; e++) {
        int f = e << 2; // Shift left by 2 is equivalent to multiplying by 4
        if (f + 3 < d * 4) { // Check to prevent out-of-bounds access
            c[e] = a[f] * b[f] + a[f + 1] * b[f + 1] + a[f + 2] * b[f + 2] + a[f + 3] * b[f + 3];
        }
    }
}

int main() {
    int d = 4; // Number of blocks
    float a[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float b[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    float c[4] = {0};

    dot_product(a, b, c, d);

    // Print the result to verify
    for (int i = 0; i < d; i++) {
        printf("c[%d] = %f\n", i, c[i]);
    }

    return 0;
}
