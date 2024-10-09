#include <stdio.h>

void add_arrays(double* a, double* b, int c, int d, int e) {
    int f, g, h, i;
    // Assuming the arrays are 4-dimensional based on the indices used: a[e][d][c][5]

    // Transform flat index into 4D index
    for (h = 1; h < e-1; h++) {
        for (g = 1; g < d-1; g++) {
            for (f = 1; f < c-1; f++) {
                for (i = 0; i < 5; i++) {
                    // Assuming row-major order for simplicity in indexing:
                    a[((h * d + g) * c + f) * 5 + i] += b[((h * d + g) * c + f) * 5 + i];
                }
            }
        }
    }
}

int main() {
    // Example usage of add_arrays
    int c = 10, d = 10, e = 10; // Example dimensions
    double a[5000], b[5000]; // Example flat array storage for 4D arrays with dimension [10][10][10][5]

    // Initialize arrays for demonstration purposes
    for (int i = 0; i < 5000; i++) {
        a[i] = i * 0.1;
        b[i] = i * 0.2;
    }

    add_arrays(a, b, c, d, e);

    // Print some results to verify
    for (int i = 0; i < 50; i++) {
        printf("%f ", a[i]);
        if ((i + 1) % 10 == 0) printf("\n");
    }

    return 0;
}
