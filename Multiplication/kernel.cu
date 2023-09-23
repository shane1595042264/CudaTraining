#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdlib>  // For the rand() function
const int M = 512;  // Number of rows in matrix A
const int N = 512;  // Number of columns in matrix A (and rows in matrix B)
const int P = 512;  // Number of columns in matrix B
__global__ void matrixMul(float* A, float* B, float* C, int M, int N, int P) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;
    if (row < M && col < P) {
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * P + col];
        }
        C[row * P + col] = sum;
    }
}
int main() {
    float* A_host, * B_host, * C_host;          // Matrices on the host
    float* A_device, * B_device, * C_device;    // Matrices on the device

    // Allocate host memory and fill matrices A and B with random values
    A_host = new float[M * N];
    B_host = new float[N * P];
    C_host = new float[M * P];

    for (int i = 0; i < M * N; i++) {
        A_host[i] = static_cast<float>(rand() % 100);
    }

    for (int i = 0; i < N * P; i++) {
        B_host[i] = static_cast<float>(rand() % 100);
    }

    // Allocate device memory
    cudaMalloc(&A_device, M * N * sizeof(float));
    cudaMalloc(&B_device, N * P * sizeof(float));
    cudaMalloc(&C_device, M * P * sizeof(float));

    // Copy matrices A and B to the device
    cudaMemcpy(A_device, A_host, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B_host, N * P * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((P + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the matrix multiplication kernel
    matrixMul << <blocksPerGrid, threadsPerBlock >> > (A_device, B_device, C_device, M, N, P);

    // Copy the resulting matrix C back to the host
    cudaMemcpy(C_host, C_device, M * P * sizeof(float), cudaMemcpyDeviceToHost);

    // Print some elements of matrix C for verification
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            std::cout << C_host[i * P + j] << " ";
        }
        std::cout << std::endl;
    }

    // Cleanup
    delete[] A_host;
    delete[] B_host;
    delete[] C_host;

    cudaFree(A_device);
    cudaFree(B_device);
    cudaFree(C_device);

    return 0;
}
