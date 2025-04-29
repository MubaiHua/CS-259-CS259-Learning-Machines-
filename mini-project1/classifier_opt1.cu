#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

bool verify(const float *ref, const half *gpu_half, size_t N, float tolerance = 0.5) {
    float *gpu_float = (float *)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; ++i) {
        gpu_float[i] = __half2float(gpu_half[i]);
    }

    size_t errors = 0;
    for (size_t i = 0; i < N; ++i) {
        float diff = fabsf(ref[i] - gpu_float[i]);
        bool error_cond = diff > tolerance && diff > fabsf(ref[i] * tolerance);
        if (ref[i] == 0.0f) error_cond = diff > tolerance;

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

void classifier_cpu(float *output, const float *input, const float *weights,
                    int B, int Ni, int Nn) {
    for (int b = 0; b < B; ++b) {
        for (int nn = 0; nn < Nn; ++nn) {
            float sum = 0.0f;
            for (int ni = 0; ni < Ni; ++ni) {
                size_t input_idx = (size_t)b * Ni + ni;
                size_t weight_idx = (size_t)nn * Ni + ni;  // Weights[Nn x Ni] layout
                sum += input[input_idx] * weights[weight_idx];
            }
            size_t output_idx = (size_t)b * Nn + nn;
            output[output_idx] = sum;
        }
    }
}

__global__ void classifier_kernel_fp16(
    half *output, const half *input, const half *weights,
    int B, int Ni, int Nn) {
    int nn = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;

    if (nn < Nn && b < B) {
        half sum_fp32 = 0.0f;

        for (int ni = 0; ni < Ni; ++ni) {
            size_t input_idx = (size_t)b * Ni + ni;
            size_t weight_idx = (size_t)nn * Ni + ni;

            sum_fp32 += input[input_idx] * weights[weight_idx];
        }

        size_t output_idx = (size_t)b * Nn + nn;
        output[output_idx] = sum_fp32;
    }
}

// --- Main Verification Function ---
int main(int argc, char *argv[]) {
    // int Ni = 25088, Nn = 4096;
    // int B = 16;

    if (argc != 5) {
        fprintf(stderr, "Error: This program requires <Ni> <Nn> <B> <CPU verify> arguments.\n");
        return 1;
    }

    int Ni, Nn, B;

    Ni = atoi(argv[1]);
    Nn = atoi(argv[2]);
    B = atoi(argv[3]);
    bool CPU_verify = strcmp(argv[4], "true") == 0;

    printf("Ni = %d\n", Ni);
    printf("Nn = %d\n", Nn);
    printf("B  = %d\n", B);

    // Calculate sizes
    size_t input_size = (size_t)B * Ni;
    size_t weight_size = (size_t)Nn * Ni;  // Assuming Nn x Ni layout
    size_t output_size = (size_t)B * Nn;

    printf("Input: %d x %d, Weights: %d x %d, Output: %d x %d\n",
           B, Ni, Nn, Ni, B, Nn);
    printf("Total elements: Input= %zu, Weights=%zu, Output=%zu\n", input_size, weight_size, output_size);

    // Allocate Host Memory
    float *h_input_f = (float *)malloc(input_size * sizeof(float));
    float *h_weights_f = (float *)malloc(weight_size * sizeof(float));
    half *h_input_h = (half *)malloc(input_size * sizeof(half));
    half *h_weights_h = (half *)malloc(weight_size * sizeof(half));
    half *h_output_gpu_h = (half *)malloc(output_size * sizeof(half));
    float *h_output_cpu_f = (float *)malloc(output_size * sizeof(float));
    if (!h_input_f || !h_weights_f || !h_input_f || !h_weights_f || !h_output_gpu_h || !h_output_cpu_f) {
        fprintf(stderr, "Failed to allocate host memory!\n");
        return 1;
    }

    // Initialize Host Data (using fixed seed for reproducibility)
    srand(456);  // Different seed than conv2d
    for (size_t i = 0; i < input_size; ++i) {
        h_input_f[i] = (float)(rand() % 20 - 10) / 20.0f;  // Smaller range helps FP16
        h_input_h[i] = __float2half(h_input_f[i]);
    }
    for (size_t i = 0; i < weight_size; ++i) {
        h_weights_f[i] = (float)(rand() % 10 - 5) / 50.0f;  // Smaller weights/range
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
    dim3 threadsPerBlock(256);                                          // 1D block for classifier
    dim3 gridDim((Nn + threadsPerBlock.x - 1) / threadsPerBlock.x, B);  // Grid covers outputs and batch
    printf("Running Classifier GPU Kernel...\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    classifier_kernel_fp16<<<gridDim, threadsPerBlock>>>(
        d_output, d_input, d_weights, B, Ni, Nn);

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
        printf("Running Classifier CPU reference...\n");
        classifier_cpu(h_output_cpu_f, h_input_f, h_weights_f, B, Ni, Nn);
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