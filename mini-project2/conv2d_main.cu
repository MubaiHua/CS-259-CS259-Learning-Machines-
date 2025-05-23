/**********************************************************************
* Fast 2-D convolution – tiled, NCHW layout
* Compile with: nvcc -O3 -arch=sm_80 conv2d_tiled.cu
*********************************************************************/
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "conv2d_tiled.cuh"
#include "conv2d_cpu.cuh"

#define TILE_SIZE 8

// --- Main Verification Function ---
int main(int argc, char *argv[]) {
    // int Nx = 224, Ny = 224, Ni = 64, Nn = 64;
    // int B = 16;
    // Kx, Ky are fixed for loop-unrolling
    if (argc != 7) {
        fprintf(stderr, "Error: This program requires <Nx> <Ny> <Ni> <Nn> <B> <CPU verify> arguments.\n");
        return 1;
    }

    int Nx, Ny, Ni, Nn, B;
    const int Kx = 3, Ky = 3;

    Nx = atoi(argv[1]);
    Ny = atoi(argv[2]);
    Ni = atoi(argv[3]);
    Nn = atoi(argv[4]);
    B = atoi(argv[5]);
    bool CPU_verify = strcmp(argv[6], "true") == 0;

    printf("Nx = %d\n", Nx);
    printf("Ny = %d\n", Ny);
    printf("Ni = %d\n", Ni);
    printf("Nn = %d\n", Nn);
    printf("B  = %d\n", B);

    int Ox = Nx - Kx + 1;
    int Oy = Ny - Ky + 1;

    // Calculate sizes
    size_t input_size = (size_t)B * Ni * Ny * Nx;
    size_t weight_size = (size_t)Nn * Ni * Ky * Kx;
    size_t output_size = (size_t)B * Nn * Oy * Ox;

    printf("Input: %d x %d x %d x %d, Weights: %d x %d x %d x %d, Output: %d x %d x %d x %d\n",
           B, Ni, Ny, Nx, Nn, Ni, Ky, Kx, B, Nn, Oy, Ox);
    printf("Total elements: Input= %zu, Weights=%zu, Output=%zu\n", input_size, weight_size, output_size);

    // Allocate Host Memory
    float *h_input = (float *)malloc(input_size * sizeof(float));
    float *h_weights = (float *)malloc(weight_size * sizeof(float));
    float *h_output_gpu = (float *)malloc(output_size * sizeof(float));
    float *h_output_cpu = (float *)malloc(output_size * sizeof(float));

    if (!h_input || !h_weights || !h_output_gpu || !h_output_cpu) {
        fprintf(stderr, "Failed to allocate host memory!\n");
        return 1;
    }

    // Initialize Host Data (using fixed seed for reproducibility)
    srand(123);
    for (size_t i = 0; i < input_size; ++i)
        h_input[i] = (float)(rand() % 20 - 10) / 10.0f;
    for (size_t i = 0; i < weight_size; ++i)
        h_weights[i] = (float)(rand() % 10 - 5) / 10.0f;

    // Allocate Device Memory
    float *d_input, *d_weights, *d_output;
    cudaMalloc((void **)&d_input, input_size * sizeof(float));
    cudaMalloc((void **)&d_weights, weight_size * sizeof(float));
    cudaMalloc((void **)&d_output, output_size * sizeof(float));

    // Copy Data Host -> Device
    printf("Copying data to GPU...\n");
    cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, weight_size * sizeof(float), cudaMemcpyHostToDevice);
    printf("Copying done.\n");

    // --- GPU Execution ---
    printf("Running Conv2D GPU Kernel...\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    const dim3 block(TILE_SIZE, TILE_SIZE);         // 128 threads
    const int TILE_OX = TILE_SIZE, TILE_OY = TILE_SIZE, TILE_NN = TILE_SIZE, TILE_NI = TILE_SIZE;
    const int IN_TILE_SIZE  = (TILE_OY + Ky - 1) * (TILE_OX + Kx - 1);
    const int W_TILE_SIZE   = TILE_NN * TILE_NI * Ky * Kx;
    size_t smem_bytes =
      (IN_TILE_SIZE + W_TILE_SIZE) * sizeof(float);
    dim3 grid(
        (Ox + TILE_OX - 1)/TILE_OX,
        (Oy + TILE_OY - 1)/TILE_OY,
        B * ((Nn + TILE_NN - 1) / TILE_NN)
    );

    conv2d_tiled<TILE_OY, TILE_OX, TILE_NN, TILE_NI, Kx, Ky>
        <<<grid, block, smem_bytes>>>(d_output, d_input, d_weights, B, Nx, Ny, Ni, Nn);


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);  // wait for kernel to finish

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU Kernel finished in %.3f ms.\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("GPU Kernel finished.\n");

    // Copy Result Device -> Host
    printf("Copying result from GPU...\n");
    cudaMemcpy(h_output_gpu, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Copying done.\n");

    if (CPU_verify) {
        // --- CPU Execution ---
        printf("Running Conv2D CPU reference...\n");
        conv2d_cpu(h_output_cpu, h_input, h_weights, B, Nx, Ny, Kx, Ky, Ni, Nn);
        printf("CPU reference finished.\n");

        // --- Verification ---
        printf("Verifying results...\n");
        verify(h_output_cpu, h_output_gpu, output_size);
    }

    // --- Cleanup ---
    printf("Cleaning up memory...\n");
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);
    free(h_input);
    free(h_weights);
    free(h_output_gpu);
    free(h_output_cpu);
    printf("Cleanup done.\n");

    return 0;
}