# PCA: EXP-1  SUM ARRAY GPU
<h3>NAME   : HARSHITHA V</h3>
<h3>REG NO : 212223230074</h3>
<h3>EX. NO : 1</h3>
<h3>DATE   : 18-10-2024</h3>
<h1> <align=center> SUM ARRAY ON HOST AND DEVICE </h3>
PCA-GPU-based-vector-summation.-Explore-the-differences.
i) Using the program sumArraysOnGPU-timer.cu, set the block.x = 1023. Recompile and run it. Compare the result with the execution configuration of block.x = 1024. Try to explain the difference and the reason.

ii) Refer to sumArraysOnGPU-timer.cu, and let block.x = 256. Make a new kernel to let each thread handle two elements. Compare the results with other execution confi gurations.
## AIM:

To perform vector addition on host and device.

## EQUIPMENTS REQUIRED:
Hardware – PCs with NVIDIA GPU & CUDA NVCC
Google Colab with NVCC Compiler




## PROCEDURE:

1. Initialize the device and set the device properties.
2. Allocate memory on the host for input and output arrays.
3. Initialize input arrays with random values on the host.
4. Allocate memory on the device for input and output arrays, and copy input data from host to device.
5. Launch a CUDA kernel to perform vector addition on the device.
6. Copy output data from the device to the host and verify the results against the host's sequential vector addition. Free memory on the host and the device.

## PROGRAM:
```
cuda_code = """
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define N 1024  // Size of the arrays

__global__ void sumArraysOnGPU(int *a, int *b, int *c) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

void initialData(int *data, int size) {
    time_t t;
    time(&t);
    srand(t);
    for (int i = 0; i < size; i++) {
        data[i] = rand() % 100; // Random values between 0 and 99
    }
}

int main() {
    int *a, *b, *c;           // Host pointers
    int *d_a, *d_b, *d_c;     // Device pointers
    size_t size = N * sizeof(int);

    // Allocate memory on the host
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    // Initialize input arrays with random values
    initialData(a, N);
    initialData(b, N);

    // Initialize output array to zero
    memset(c, 0, size);

    // Allocate memory on the device
    cudaError_t status;
    status = cudaMalloc((void **)&d_a, size);
    if (status != cudaSuccess) {
        printf("cudaMalloc failed for d_a: %s\\n", cudaGetErrorString(status));
        return 0;
    }

    status = cudaMalloc((void **)&d_b, size);
    if (status != cudaSuccess) {
        printf("cudaMalloc failed for d_b: %s\\n", cudaGetErrorString(status));
        return 0;
    }

    status = cudaMalloc((void **)&d_c, size);
    if (status != cudaSuccess) {
        printf("cudaMalloc failed for d_c: %s\\n", cudaGetErrorString(status));
        return 0;
    }

    // Copy input data from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch the kernel
    int blockSize = 256; // Number of threads in each block
    int numBlocks = (N + blockSize - 1) / blockSize; // Calculate number of blocks
    sumArraysOnGPU<<<numBlocks, blockSize>>>(d_a, d_b, d_c);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\\n", cudaGetErrorString(err));
        return 0;
    }

    // Copy output data from device to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Check for errors
    for (int i = 0; i < N; i++) {
        if (c[i] != (a[i] + b[i])) {
            printf("Error at index %d: %d + %d != %d\\n", i, a[i], b[i], c[i]);
            return 0;
        }
    }

    printf("Test passed!\\n");

    // Free memory on the device and host
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);

    return 0;
}
"""

with open("sumArraysOnGPU.cu", "w") as f:
    f.write(cuda_code)

```
## OUTPUT:
![image](https://github.com/user-attachments/assets/ee3f71b7-8f0a-4bfe-9399-e19803e2a2dc)


## RESULT:
Thus, Implementation of sum arrays on host and device is done in nvcc cuda using random number.
