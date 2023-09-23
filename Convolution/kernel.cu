#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdlib>  // For the rand() function

const int IMAGE_WIDTH = 512;
const int IMAGE_HEIGHT = 512;
const int KERNEL_SIZE = 3;  // This is for a 3x3 kernel

__global__ void convolution2D(float* I, float* K, float* O, int width, int height, int kSize) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float value = 0.0;

    if (row < height && col < width) {
        for (int i = -kSize / 2; i <= kSize / 2; i++) {
            for (int j = -kSize / 2; j <= kSize / 2; j++) {
                int curRow = row + i;
                int curCol = col + j;

                // Check boundary
                if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                    value += I[curRow * width + curCol] * K[(i + kSize / 2) * kSize + (j + kSize / 2)];
                }
            }
        }
        O[row * width + col] = value;
    }
}
int main() {
    float* I_host, * K_host, * O_host;          // Image, kernel, and output on the host
    float* I_device, * K_device, * O_device;    // Image, kernel, and output on the device

    // Allocate host memory and initialize image and kernel
    I_host = new float[IMAGE_WIDTH * IMAGE_HEIGHT];
    K_host = new float[KERNEL_SIZE * KERNEL_SIZE];
    O_host = new float[IMAGE_WIDTH * IMAGE_HEIGHT];

    // Here, fill I_host with some values (e.g., random or sample image data)
    // For simplicity, I just fill some random grayscale values for these values. 
    for (int i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; i++) {
        I_host[i] = static_cast<float>(rand() % 256);
    }

    // And fill K_host with your desired kernel values
    // It's an edge detection filter.
    float edge_detection_kernel[3][3] = {
    {-1, -1, -1},
    {-1, 8, -1},
    {-1, -1, -1}
    };

    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            K_host[i * KERNEL_SIZE + j] = edge_detection_kernel[i][j];
        }
    }

    // Allocate device memory
    cudaMalloc(&I_device, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(float));
    cudaMalloc(&K_device, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    cudaMalloc(&O_device, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(float));

    // Copy image and kernel to the device
    cudaMemcpy(I_device, I_host, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(K_device, K_host, KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((IMAGE_WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (IMAGE_HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the convolution2D kernel
    convolution2D <<<blocksPerGrid, threadsPerBlock >>> (I_device, K_device, O_device, IMAGE_WIDTH, IMAGE_HEIGHT, KERNEL_SIZE);

    // Copy the resulting image back to the host
    cudaMemcpy(O_host, O_device, IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    delete[] I_host;
    delete[] K_host;
    delete[] O_host;

    cudaFree(I_device);
    cudaFree(K_device);
    cudaFree(O_device);

    return 0;
}
