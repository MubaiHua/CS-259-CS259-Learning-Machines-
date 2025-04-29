#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_H 16  // Thread block height (output tile height)
#define BLOCK_W 16  // Thread block width (output tile width)
#define TILE_Ni 16  // Input channels processed per iteration
#define TILE_Nn 16  // Output channels computed per block iteration/thread accumulation
#define K_MAX 3

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

__global__ void conv2d_kernel_nhwc_fp16_shared_memory(
    half *output, const half *input, const half *weights,
    int B, int Nx, int Ny, int Kx, int Ky, int Ni, int Nn) {
    int Oy = Ny - Ky + 1;
    int Ox = Nx - Kx + 1;

    // Shared memory allocation
    // Input tile needs halo: (BLOCK_H + Ky - 1) x (BLOCK_W + Kx - 1) x TILE_Ni
    // Weight tile: Ky x Kx x TILE_Ni x TILE_Nn
    // Use template or larger fixed size if Kx/Ky vary significantly
    const int INPUT_TILE_H_PADDED = BLOCK_H + K_MAX - 1;
    const int INPUT_TILE_W_PADDED = BLOCK_W + K_MAX - 1;
    __shared__ half input_tile[INPUT_TILE_H_PADDED * INPUT_TILE_W_PADDED * TILE_Ni];
    __shared__ half weight_tile[K_MAX * K_MAX * TILE_Ni * TILE_Nn];  // Use K_MAX

    // Thread indices within the block
    const int tx = threadIdx.x;  // 0..BLOCK_W-1
    const int ty = threadIdx.y;  // 0..BLOCK_H-1
    const int thread_id_in_block = ty * BLOCK_W + tx;
    const int block_size = BLOCK_H * BLOCK_W;  // Total threads per block

    // Block indices mapping - adjusted for channel tiling
    const int block_ox_base = blockIdx.x * BLOCK_W;  // Base output X for this block
    const int block_oy_base = blockIdx.y * BLOCK_H;  // Base output Y for this block
    // Calculate base output channel and batch index from blockIdx.z
    const int num_nn_tiles = (Nn + TILE_Nn - 1) / TILE_Nn;
    const int z_idx = blockIdx.z;
    const int b = z_idx / num_nn_tiles;  // Batch index for this block
    const int nn_tile_idx = z_idx % num_nn_tiles;
    const int block_nn_base = nn_tile_idx * TILE_Nn;  // Base output channel for this block

    // Target output pixel calculated by this thread
    const int ox = block_ox_base + tx;
    const int oy = block_oy_base + ty;

    // Accumulators (one per output channel tile handled by this thread)
    half accumulators[TILE_Nn];
    for (int i = 0; i < TILE_Nn; ++i) {
        accumulators[i] = 0.0f;
    }

    // Check if block is valid for the batch dimension early exit
    if (b >= B) {
        return;
    }

    // Calculate start coordinates for loading the input tile's top-left corner (global indices)
    const int input_tile_y_global_start = block_oy_base;  // Assumes Stride=1, Padding=0
    const int input_tile_x_global_start = block_ox_base;  // Assumes Stride=1, Padding=0

    // Loop over input channel tiles (ni_base)
    for (int ni_base = 0; ni_base < Ni; ni_base += TILE_Ni) {
        // --- Cooperatively Load Input Tile to Shared Memory ---
        // Each thread loads multiple elements to fill the shared tile
        for (int load_idx = thread_id_in_block;
             load_idx < INPUT_TILE_H_PADDED * INPUT_TILE_W_PADDED * TILE_Ni;
             load_idx += block_size) {
            // Deconstruct load_idx into (y_offset, x_offset, ni_offset) within the shared tile
            int ni_offset = load_idx % TILE_Ni;
            int plane_idx = load_idx / TILE_Ni;
            int x_offset = plane_idx % INPUT_TILE_W_PADDED;
            int y_offset = plane_idx / INPUT_TILE_W_PADDED;

            // Calculate global coordinates corresponding to this shared memory element
            int current_iy = input_tile_y_global_start + y_offset;
            int current_ix = input_tile_x_global_start + x_offset;
            int current_ni = ni_base + ni_offset;

            // Calculate destination index in shared memory
            // Layout: [y_offset, x_offset, ni_offset]
            int shared_idx = y_offset * INPUT_TILE_W_PADDED * TILE_Ni + x_offset * TILE_Ni + ni_offset;

            // Load from global memory if within bounds, otherwise pad with zero
            if (current_ni < Ni && current_iy >= 0 && current_iy < Ny && current_ix >= 0 && current_ix < Nx) {
                // Global Input Index: [b, current_iy, current_ix, current_ni]
                size_t input_idx_global = (size_t)b * Ny * Nx * Ni +
                                          (size_t)current_iy * Nx * Ni +
                                          (size_t)current_ix * Ni +
                                          current_ni;
                input_tile[shared_idx] = input[input_idx_global];
            } else {
                input_tile[shared_idx] = 0.0f;  // Zero padding
            }
        }

        // --- Cooperatively Load Weight Tile to Shared Memory ---
        // Load weights for the current output channel tile (block_nn_base) and input channel tile (ni_base)
        const int weight_tile_size = K_MAX * K_MAX * TILE_Ni * TILE_Nn;
        for (int load_idx = thread_id_in_block;
             load_idx < weight_tile_size;
             load_idx += block_size) {
            // Deconstruct load_idx to find (nn_offset, ky, kx, ni_offset) within the weight tile
            // Layout: [nn_offset, ky, kx, ni_offset]
            int nn_offset = (load_idx / (K_MAX * K_MAX * TILE_Ni));  // Offset within TILE_Nn
            int kernel_plane_idx = load_idx % (K_MAX * K_MAX * TILE_Ni);
            int ni_offset = kernel_plane_idx % TILE_Ni;  // Offset within TILE_Ni
            int kernel_idx = kernel_plane_idx / TILE_Ni;
            int kx = kernel_idx % K_MAX;
            int ky = kernel_idx / K_MAX;

            // Only load if ky < Ky and kx < Kx (actual runtime dimensions)
            if (ky < Ky && kx < Kx) {
                // Calculate global indices using actual ky, kx
                int current_nn = block_nn_base + nn_offset;
                int current_ni = ni_base + ni_offset;

                // Calculate destination index in shared memory using K_MAX based layout
                int shared_idx = nn_offset * K_MAX * K_MAX * TILE_Ni + ky * K_MAX * TILE_Ni + kx * TILE_Ni + ni_offset;

                // Load from global memory if within bounds, otherwise pad
                if (current_nn < Nn && current_ni < Ni) {
                    // Global Weight Index: [current_nn, ky, kx, current_ni] (using actual Ky, Kx)
                    size_t weight_idx_global = (size_t)current_nn * Ky * Kx * Ni +
                                               (size_t)ky * Kx * Ni +
                                               (size_t)kx * Ni +
                                               current_ni;
                    weight_tile[shared_idx] = weights[weight_idx_global];
                } else {
                    weight_tile[shared_idx] = 0.0f;  // Zero padding
                }
            }
        }

        // Synchronize: Ensure all loads into shared memory are complete before computation
        __syncthreads();

        // --- Perform Convolution using Shared Memory Data ---
        // Check if the output pixel calculated by this thread is valid
        if (ox < Ox && oy < Oy) {
            // Loop over the output channels this block computes (TILE_Nn)
            for (int nn_offset = 0; nn_offset < TILE_Nn; ++nn_offset) {
                int current_nn = block_nn_base + nn_offset;
                // Check if this specific output channel is needed (handles Nn not multiple of TILE_Nn)
                if (current_nn < Nn) {
                    // Loop over kernel dimensions
                    for (int ky = 0; ky < Ky; ++ky) {
                        for (int kx = 0; kx < Kx; ++kx) {
                            // Loop over input channels in the current tile (TILE_Ni)
                            #pragma unroll  // Suggest unrolling the innermost loop if TILE_Ni is small and fixed
                            for (int ni_offset = 0; ni_offset < TILE_Ni; ++ni_offset) {
                                // Calculate index into the shared input tile
                                // Coordinates relative to the top-left of the padded shared input tile
                                int shared_input_y = ty + ky;  // ty is thread's y offset within output tile
                                int shared_input_x = tx + kx;  // tx is thread's x offset within output tile
                                // Shared Input Index: [shared_input_y, shared_input_x, ni_offset]
                                int input_shared_idx = shared_input_y * INPUT_TILE_W_PADDED * TILE_Ni +
                                                       shared_input_x * TILE_Ni +
                                                       ni_offset;

                                // Calculate index into the shared weight tile
                                // Shared Weight Index: [nn_offset, ky, kx, ni_offset]
                                int weight_shared_idx = nn_offset * Ky * Kx * TILE_Ni +
                                                        ky * Kx * TILE_Ni +
                                                        kx * TILE_Ni +
                                                        ni_offset;

                                // Accumulate product (reading from shared memory)
                                accumulators[nn_offset] += input_tile[input_shared_idx] * weight_tile[weight_shared_idx];
                            }  // ni_offset
                        }  // kx
                    }  // ky
                }  // if current_nn < Nn
            }  // nn_offset
        }  // if output pixel is valid

        // Synchronize: Ensure all computations using the current tiles are done
        // before potentially loading the next tiles in the ni_base loop
        __syncthreads();

    }  // End loop over ni_base

    // --- Write results to Global Memory ---
    // Check if the output pixel is valid before writing
    if (ox < Ox && oy < Oy) {
        // Loop over the output channels computed by this thread
        for (int nn_offset = 0; nn_offset < TILE_Nn; ++nn_offset) {
            int current_nn = block_nn_base + nn_offset;
            // Check if this specific output channel is needed
            if (current_nn < Nn) {
                // Global Output Index: [b, oy, ox, current_nn]
                size_t output_idx_global = (size_t)b * Oy * Ox * Nn +
                                           (size_t)oy * Ox * Nn +
                                           (size_t)ox * Nn +
                                           current_nn;
                // Convert final accumulated float back to half for storing
                output[output_idx_global] = accumulators[nn_offset];
            }
        }
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
    dim3 threadsPerBlock(BLOCK_W, BLOCK_H);

    int gridX = (Ox + BLOCK_W - 1) / BLOCK_W;
    int gridY = (Oy + BLOCK_H - 1) / BLOCK_H;
    int num_nn_tiles = (Nn + TILE_Nn - 1) / TILE_Nn;
    int gridZ = B * num_nn_tiles;  // Total blocks needed in Z dimension

    dim3 gridDim(gridX, gridY, gridZ);

    printf("Running Conv2D GPU Kernel...\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    conv2d_kernel_nhwc_fp16_shared_memory<<<gridDim, threadsPerBlock>>>(
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