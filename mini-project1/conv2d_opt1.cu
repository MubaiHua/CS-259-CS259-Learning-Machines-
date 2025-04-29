#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// --- Verification Function ---
bool verify(const float *ref, const half *gpu_half, size_t N, float tolerance = 0.5) {
    float *gpu_float = (float *)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; ++i) {
        gpu_float[i] = __half2float(gpu_half[i]);
    }

    size_t errors = 0;
    for (size_t i = 0; i < N; ++i) {
        float diff = fabsf(ref[i] - gpu_float[i]);
        bool error_cond = diff > tolerance && diff > fabsf(ref[i] * tolerance);
        if (ref[i] == 0.0f)
            error_cond = diff > tolerance;

        if (error_cond) {
            if (errors < 10) {
                fprintf(stderr, " Verification failed at index %zu: Ref=%.6f, GPU(FP16)=%.6f, Diff=%.6f\n", i, ref[i], gpu_float[i], diff);
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

void conv2d_cpu_nhwc_fp32_ref(float *output, const float *input, const float *weights,
                              int B, int Nx, int Ny, int Kx, int Ky, int Ni, int Nn) {
    int Oy = Ny - Ky + 1;
    int Ox = Nx - Kx + 1;
    for (int b = 0; b < B; ++b) {
        for (int oy = 0; oy < Oy; ++oy) {
            for (int ox = 0; ox < Ox; ++ox) {
                for (int nn = 0; nn < Nn; ++nn) {
                    float sum = 0.0f;
                    for (int ky = 0; ky < Ky; ++ky) {
                        for (int kx = 0; kx < Kx; ++kx) {
                            for (int ni = 0; ni < Ni; ++ni) {
                                int iy = oy + ky;
                                int ix = ox + kx;
                                size_t input_idx = (size_t)b * Ny * Nx * Ni + (size_t)iy * Nx * Ni + (size_t)ix * Ni + ni;
                                size_t weight_idx = (size_t)nn * Ky * Kx * Ni + (size_t)ky * Kx * Ni + (size_t)kx * Ni + ni;
                                if (iy >= 0 && iy < Ny && ix >= 0 && ix < Nx) {
                                    sum += input[input_idx] * weights[weight_idx];
                                }
                            }
                        }
                    }
                    size_t output_idx = (size_t)b * Oy * Ox * Nn + (size_t)oy * Ox * Nn + (size_t)ox * Nn + nn;
                    output[output_idx] = sum;  // Keep FP32 result for reference
                }
            }
        }
    }
}

__global__ void conv2d_kernel_nhwc_fp16(
    half *output, const half *input, const half *weights,
    int B, int Nx, int Ny, int Kx, int Ky, int Ni, int Nn) {
    int Oy = Ny - Ky + 1;
    int Ox = Nx - Kx + 1;
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int nn = blockIdx.z % Nn;
    int b = blockIdx.z / Nn;

    if (ox < Ox && oy < Oy && b < B) {
        half acc_fp16 = 0.0f;  // Accumulate in float
        for (int ky = 0; ky < Ky; ++ky) {
            for (int kx = 0; kx < Kx; ++kx) {
                for (int ni = 0; ni < Ni; ++ni) {
                    int iy = oy + ky;
                    int ix = ox + kx;
                    if (iy >= 0 && iy < Ny && ix >= 0 && ix < Nx) {
                        size_t input_idx = (size_t)b * Ny * Nx * Ni + (size_t)iy * Nx * Ni + (size_t)ix * Ni + ni;
                        size_t weight_idx = (size_t)nn * Ky * Kx * Ni + (size_t)ky * Kx * Ni + (size_t)kx * Ni + ni;
                        acc_fp16 += input[input_idx] * weights[weight_idx];
                    }
                }
            }
        }
        size_t output_idx = (size_t)b * Oy * Ox * Nn + (size_t)oy * Ox * Nn + (size_t)ox * Nn + nn;
        output[output_idx] = acc_fp16;  // Convert back to half
    }
}

// --- Main Verification Function ---
int main(int argc, char *argv[]) {
    // int Nx = 224, Ny = 224, Kx = 3, Ky = 3, Ni = 64, Nn = 64;
    // int B = 16;

    if (argc != 9) {
        fprintf(stderr, "Error: This program requires <Nx> <Ny> <Kx> <Ky> <Ni> <Nn> <B> <CPU verify> arguments.\n");
        return 1;
    }

    int Nx, Ny, Kx, Ky, Ni, Nn, B;

    Nx = atoi(argv[1]);
    Ny = atoi(argv[2]);
    Kx = atoi(argv[3]);
    Ky = atoi(argv[4]);
    Ni = atoi(argv[5]);
    Nn = atoi(argv[6]);
    B = atoi(argv[7]);
    bool CPU_verify = strcmp(argv[8], "true") == 0;

    printf("Nx = %d\n", Nx);
    printf("Ny = %d\n", Ny);
    printf("Kx = %d\n", Kx);
    printf("Ky = %d\n", Ky);
    printf("Ni = %d\n", Ni);
    printf("Nn = %d\n", Nn);
    printf("B  = %d\n", B);

    int Ox = Nx - Kx + 1;
    int Oy = Ny - Ky + 1;

    // Calculate sizes
    size_t input_size = (size_t)B * Ny * Nx * Ni;
    size_t weight_size = (size_t)Nn * Ky * Kx * Ni;
    size_t output_size = (size_t)B * Oy * Ox * Nn;

    printf("Input: %d x %d x %d x %d, Weights: %d x %d x %d x %d, Output: %d x %d x %d x %d\n",
           B, Ni, Ny, Nx, Nn, Ni, Ky, Kx, B, Nn, Oy, Ox);
    printf("Total elements: Input= %zu, Weights=%zu, Output=%zu\n", input_size, weight_size, output_size);

    // Allocate Host Memory
    float *h_input_f = (float *)malloc(input_size * sizeof(float));
    float *h_weights_f = (float *)malloc(weight_size * sizeof(float));
    half *h_input_h = (half *)malloc(input_size * sizeof(half));
    half *h_weights_h = (half *)malloc(weight_size * sizeof(half));
    float *h_output_cpu_f = (float *)malloc(output_size * sizeof(float));
    half *h_output_gpu_h = (half *)malloc(output_size * sizeof(half));

    if (!h_input_f || !h_weights_f || !h_input_h || !h_weights_h || !h_output_cpu_f || !h_output_gpu_h) {
        fprintf(stderr, "Failed to allocate host memory!\n");
        return 1;
    }

    // Initialize Host Data (using fixed seed for reproducibility)
    srand(123);
    for (size_t i = 0; i < input_size; ++i) {
        h_input_f[i] = (float)(rand() % 20 - 10) / 20.0f;  // Smaller range helps FP16
        h_input_h[i] = __float2half(h_input_f[i]);
    }
    for (size_t i = 0; i < weight_size; ++i) {
        h_weights_f[i] = (float)(rand() % 10 - 5) / 20.0f;  // Smaller range helps FP16
        h_weights_h[i] = __float2half(h_weights_f[i]);
    }

    // Allocate Device Memory
    half *d_input, *d_weights, *d_output;
    cudaMalloc((void **)&d_input, input_size * sizeof(half));
    cudaMalloc((void **)&d_weights, weight_size * sizeof(half));
    cudaMalloc((void **)&d_output, output_size * sizeof(half));

    // Copy Data Host -> Device
    printf("Copying data to GPU...\n");
    cudaMemcpy(d_input, h_input_h, input_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights_h, weight_size * sizeof(half), cudaMemcpyHostToDevice);
    printf("Copying done.\n");

    // --- GPU Execution ---
    dim3 threadsPerBlock(16, 16);
    dim3 gridDim((Ox + threadsPerBlock.x - 1) / threadsPerBlock.x,
                 (Oy + threadsPerBlock.y - 1) / threadsPerBlock.y,
                 B * Nn);
    printf("Running Conv2D GPU Kernel...\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    conv2d_kernel_nhwc_fp16<<<gridDim, threadsPerBlock>>>(
        d_output, d_input, d_weights, B, Nx, Ny, Kx, Ky, Ni, Nn);

    cudaEventRecord(stop);

    cudaGetLastError();
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU Kernel finished in %.3f ms.\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("GPU Kernel finished.\n");

    // Copy Result Device -> Host
    printf("Copying result from GPU...\n");
    cudaMemcpy(h_output_gpu_h, d_output, output_size * sizeof(half), cudaMemcpyDeviceToHost);
    printf("Copying done.\n");

    if (CPU_verify) {
        // --- CPU Execution ---
        printf("Running Conv2D CPU reference...\n");
        conv2d_cpu_nhwc_fp32_ref(h_output_cpu_f, h_input_f, h_weights_f, B, Nx, Ny, Kx, Ky, Ni, Nn);
        printf("CPU reference finished.\n");

        // --- Verification ---
        printf("Verifying results...\n");
        verify(h_output_cpu_f, h_output_gpu_h, output_size);
    }

    // --- Cleanup ---
    printf("Cleaning up memory...\n");
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);
    free(h_input_f);
    free(h_weights_f);
    free(h_input_h);
    free(h_weights_h);
    free(h_output_gpu_h);
    free(h_output_cpu_f);
    printf("Cleanup done.\n");

    return 0;
}