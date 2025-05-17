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



void conv2d_cpu(float *output, const float *input, const float *weights,
                int B, int Nx, int Ny, int Kx, int Ky, int Ni, int Nn) {
    int Oy = Ny - Ky + 1;
    int Ox = Nx - Kx + 1;
    // Simpler loops for clarity in basic verification
    for (int b = 0; b < B; ++b) {
        for (int nn = 0; nn < Nn; ++nn) {
            for (int oy = 0; oy < Oy; ++oy) {
                for (int ox = 0; ox < Ox; ++ox) {
                    float sum = 0.0f;
                    for (int ni = 0; ni < Ni; ++ni) {
                        for (int ky = 0; ky < Ky; ++ky) {
                            for (int kx = 0; kx < Kx; ++kx) {
                                int iy = oy + ky;
                                int ix = ox + kx;
                                size_t input_idx = (size_t)b * Ni * Ny * Nx + (size_t)ni * Ny * Nx + (size_t)iy * Nx + ix;
                                size_t weight_idx = (size_t)nn * Ni * Ky * Kx + (size_t)ni * Ky * Kx + (size_t)ky * Kx + kx;
                                sum += input[input_idx] * weights[weight_idx];
                            }
                        }
                    }
                    size_t output_idx = (size_t)b * Nn * Oy * Ox + (size_t)nn * Oy * Ox + (size_t)oy * Ox + ox;
                    output[output_idx] = sum;
                }
            }
        }
    }
}

// --- Verification Function ---
bool verify(const float *ref, const float *gpu, size_t N, float tolerance = 1e-4) {
    size_t errors = 0;
    for (size_t i = 0; i < N; ++i) {
        if (fabs(ref[i] - gpu[i]) > tolerance) {
            if (errors < 10) {
                fprintf(stderr, "Verification failed at index %zu: Ref=%.6f, GPU=%.6f, Diff=%.6f\n", i, ref[i], gpu[i], fabs(ref[i] - gpu[i]));
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("Verification Successful!\n");
        return true;
    } else {
        printf("Verification FAILED with %zu errors!\n", errors);
        return false;
    }
}

// --- Main Verification Function ---
int main(int argc, char *argv[]) {
    // int Nx = 224, Ny = 224, Ni = 64, Nn = 64;
    // int B = 16;

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
    cudaEventRecord(start);

    const dim3 block(16, 8);         // 128 threads
    const int TILE_OX = 16, TILE_OY = 8, TILE_NN = 4, TILE_NI = 8;

    size_t smem_bytes = (TILE_OX + Kx - 1) * (TILE_OY + Ky - 1) * sizeof(float);

    dim3 grid(
        (Ox + TILE_OX - 1)/TILE_OX,
        (Oy + TILE_OY - 1)/TILE_OY,
        B * ((Nn + TILE_NN - 1) / TILE_NN)
    );

    conv2d_tiled<TILE_OY, TILE_OX, TILE_NN, TILE_NI, Kx, Ky>
        <<<grid, block, smem_bytes>>>(d_output, d_input, d_weights, B, Nx, Ny, Ni, Nn);

    cudaEventRecord(stop);
    cudaGetLastError();
    cudaDeviceSynchronize();
    // cudaEventSynchronize(stop);

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