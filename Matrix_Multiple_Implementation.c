#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>

#define N 2048         // Matrices Size
#define BLOCKSIZE 64    // Block Size
double matrixMultiResult[N][N] = {0.0};
double firstMatrix[N][N] = {0.0};
double secondMatrix[N][N] = {0.0};


void matrixMultiplication() {
#pragma omp parallel for
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            double resultValue = 0;
            #pragma omp parallel for reduction(+:resultValue)
            for (int transNumber = 0; transNumber < N; transNumber++) {
                resultValue += firstMatrix[row][transNumber] * secondMatrix[transNumber][col];
            }
            matrixMultiResult[row][col] = resultValue;
        }
    }
}

// Block-optimized matrices multiplication
void matrixMultiBlockOptimized() {
#pragma omp parallel for
    for (int sj = 0; sj < N; sj += BLOCKSIZE) {
        for (int si = 0; si < N; si += BLOCKSIZE) {
            for (int sk = 0; sk < N; sk += BLOCKSIZE) {
                for (int i = si; i < si + BLOCKSIZE; i++) {
                    for (int j = sj; j < sj + BLOCKSIZE; j++) {
                        double cij = matrixMultiResult[i][j];
                        // Inner loop for parallelization using reduction
                        #pragma omp parallel for reduction(+:cij)
                        for (int k = sk; k < sk + BLOCKSIZE; k++) {
                            cij += firstMatrix[i][k] * secondMatrix[k][j];
                        }
                        matrixMultiResult[i][j] = cij;
                    }
                }
            }
        }
    }
}

// Matrices Initialization
void matrixInit() {
#pragma omp parallel for  // Initialization of matrices through parallelism
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            srand(row + col);
            firstMatrix[row][col] = (rand() & 10) * 1.1;
            secondMatrix[row][col] = (rand() & 10) * 1.1;
        }
    }
}

int main() {
    matrixInit();
    {
        double c1 = omp_get_wtime();
        matrixMultiplication();
        double c2 = omp_get_wtime();
        printf("Task 1 - Time: %f seconds\n", c2 - c1);
    }
    {
        double c1 = omp_get_wtime();
        matrixMultiBlockOptimized();
        double c2 = omp_get_wtime();
        printf("Task 2 - Time: %f seconds\n", c2 - c1);
    }
    return 0;
}