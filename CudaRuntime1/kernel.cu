
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>



__global__ void printThreadIndices() {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("blockIdx.x: %d, threadIdx.x: %d, tid: %d\n", blockIdx.x, threadIdx.x, tid);
}


int main() {
    int threadsPerBlock = 4;
    int blocksPerGrid = 3;

    // Launch the kernel
    printThreadIndices<<<blocksPerGrid, threadsPerBlock>>>();

    // Ensure all CUDA operations are completed before proceeding
    cudaDeviceSynchronize();

    return 0;
}
