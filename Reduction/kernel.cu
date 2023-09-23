#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdlib>  // For the rand() function

const int N = 1 << 20;  // This will give us 2^20 elements, i.e., 1,048,576 elements

__global__ void reduceKernel(float* data, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check
    if (tid >= size) return;

    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            data[tid] += data[tid + s];
        }
        __syncthreads();  // Ensure all additions at the current step are done!
    }
}
int main() {
    float* dataHost;  // Data on the host
    float* dataDevice;  // Data on the device

    // Allocate host memory and fill with random numbers
    dataHost = new float[N];
    for (int i = 0; i < N; i++) {
        dataHost[i] = static_cast<float>(rand() % 100);  // Random numbers between 0 and 99
    }

    // Allocate device memory
    cudaMalloc(&dataDevice, N * sizeof(float));

    // Copy data to device
    cudaMemcpy(dataDevice, dataHost, N * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    reduceKernel <<<blocksPerGrid, threadsPerBlock >>> (dataDevice, N);

    // Copy the result back
    cudaMemcpy(dataHost, dataDevice, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the result (the sum should be in the first element)
    std::cout << "Sum of array elements: " << dataHost[0] << std::endl;

    // Cleanup
    delete[] dataHost;
    cudaFree(dataDevice);

    return 0;
}
