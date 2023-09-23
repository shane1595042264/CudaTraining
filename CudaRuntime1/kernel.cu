
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <stdio.h>


const int N = 1 << 20;  // Number of elements in our arrays
float* a, * b, * c;      // Arrays for the vectors and the result on the host
float* d_a, * d_b, * d_c;  // Arrays for the vectors and the result on the device

__global__ void printThreadIndices() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("blockIdx.x: %d, threadIdx.x: %d, tid: %d\n", blockIdx.x, threadIdx.x, tid);
}
__global__ void vectorAdd(float* a, float* b, float* c, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}


int main() {
    // Memory allocation on host
    a = new float[N];
    b = new float[N];
    c = new float[N];

    // Fill arrays with random data
    for (int i = 0; i < N; i++) {
        a[i] = rand() % 100;  // Random numbers between 0 and 99
        b[i] = rand() % 100;
    }

    // Allocate memory on the device
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);
    // Kernel Launch
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd <<<blocksPerGrid, threadsPerBlock>>> (d_a, d_b, d_c, N);
    // Copy result back to host
    cudaMemcpy(c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    // Print the first 10 results for verification
    for (int i = 0; i < 10; i++) {
        printf("a[%d] = %f, b[%d] = %f, c[%d] = %f\n", i, a[i], i, b[i], i, c[i]);
    }

    // Cleanup
    delete[] a;
    delete[] b;
    delete[] c;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);


    return 0;
}
