#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// --- Verification Function ---
bool verify(const float *ref, const float *gpu, size_t N, float tolerance = 1e-4)
{
    size_t errors = 0;
    for (size_t i = 0; i < N; ++i)
    {
        if (fabs(ref[i] - gpu[i]) > tolerance)
        {
            if (errors < 10)
            {
                fprintf(stderr, "Verification failed at index %zu: Ref=%.6f, GPU=%.6f, Diff=%.6f\n", i, ref[i], gpu[i], fabs(ref[i] - gpu[i]));
            }
            errors++;
        }
    }
    if (errors == 0)
    {
        printf("Verification Successful!\n");
        return true;
    }
    else
    {
        printf("Verification FAILED with %zu errors!\n", errors);
        return false;
    }
}

void conv2d_cpu(float *output, const float *input, const float *weights,
                int B, int Nx, int Ny, int Kx, int Ky, int Ni, int Nn)
{
    int Oy = Ny - Ky + 1;
    int Ox = Nx - Kx + 1;
    // Simpler loops for clarity in basic verification
    for (int b = 0; b < B; ++b)
    {
        for (int nn = 0; nn < Nn; ++nn)
        {
            for (int oy = 0; oy < Oy; ++oy)
            {
                for (int ox = 0; ox < Ox; ++ox)
                {
                    float sum = 0.0f;
                    for (int ni = 0; ni < Ni; ++ni)
                    {
                        for (int ky = 0; ky < Ky; ++ky)
                        {
                            for (int kx = 0; kx < Kx; ++kx)
                            {
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

__global__ void conv2d_kernel(float *output, const float *input, const float *weights,
                              int B, int Nx, int Ny, int Kx, int Ky, int Ni, int Nn)
{
    int Oy = Ny - Ky + 1;
    int Ox = Nx - Kx + 1;
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int nn = blockIdx.z % Nn;
    int b = blockIdx.z / Nn;

    if (ox < Ox && oy < Oy && b < B)
    {
        float acc = 0.0f;
        for (int ni = 0; ni < Ni; ++ni)
        {
            for (int ky = 0; ky < Ky; ++ky)
            {
                for (int kx = 0; kx < Kx; ++kx)
                {
                    int iy = oy + ky;
                    int ix = ox + kx;
                    size_t input_idx = (size_t)b * Ni * Ny * Nx + (size_t)ni * Ny * Nx + (size_t)iy * Nx + ix;
                    size_t weight_idx = (size_t)nn * Ni * Ky * Kx + (size_t)ni * Ky * Kx + (size_t)ky * Kx + kx;
                    acc += input[input_idx] * weights[weight_idx];
                }
            }
        }
        size_t output_idx = (size_t)b * Nn * Oy * Ox + (size_t)nn * Oy * Ox + (size_t)oy * Ox + ox;
        output[output_idx] = acc;
    }
}

// --- Main Verification Function ---
int main(int argc, char *argv[])
{
    // int Nx = 224, Ny = 224, Kx = 3, Ky = 3, Ni = 64, Nn = 64;
    // int B = 16;

    if (argc != 9)
    {
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

    // --- Create cuDNN Handle
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

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

    if (!h_input || !h_weights || !h_output_gpu || !h_output_cpu)
    {
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


    // --- Create cuDNN Descriptors
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    // --- Set Tensor Descriptors
    cudnnSetTensor4dDescriptor(input_desc,
                            CUDNN_TENSOR_NCHW,
                            CUDNN_DATA_FLOAT,
                            B, Ni, Ny, Nx);

    cudnnSetFilter4dDescriptor(filter_desc,
                            CUDNN_DATA_FLOAT,
                            CUDNN_TENSOR_NCHW,
                            Nn, Ni, Ky, Kx);

    cudnnSetConvolution2dDescriptor(conv_desc,
                                    0, 0,  // pad_h, pad_w (no padding)
                                    1, 1,  // stride_h, stride_w
                                    1, 1,  // dilation_h, dilation_w
                                    CUDNN_CROSS_CORRELATION,
                                    CUDNN_DATA_FLOAT);

    // --- Set Output Tensor Descriptor
    cudnnSetTensor4dDescriptor(output_desc,
                            CUDNN_TENSOR_NCHW,
                            CUDNN_DATA_FLOAT,
                            B, Nn, Oy, Ox);

    // --- Pick Fastest Convolution Algorithm
    int returnedAlgoCount = 0;
    cudnnConvolutionFwdAlgoPerf_t perf_results;

    cudnnFindConvolutionForwardAlgorithm(cudnn,
                                        input_desc,
                                        filter_desc,
                                        conv_desc,
                                        output_desc,
                                        1,
                                        &returnedAlgoCount,
                                        &perf_results);

    // --- Get Workspace Size
    size_t workspace_bytes = 0;
    cudnnConvolutionFwdAlgo_t algo = perf_results.algo;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                            input_desc,
                                            filter_desc,
                                            conv_desc,
                                            output_desc,
                                            algo,
                                            &workspace_bytes);

    // --- Allocate Workspace

    void *d_workspace = nullptr;
    if (workspace_bytes > 0)
        cudaMalloc(&d_workspace, workspace_bytes);

    // --- Launch cuDNN Convolution
    const float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(cudnn,
                            &alpha,
                            input_desc, d_input,
                            filter_desc, d_weights,
                            conv_desc, algo,
                            d_workspace, workspace_bytes,
                            &beta,
                            output_desc, d_output);

    cudaDeviceSynchronize();
    printf("cuDNN Conv2D finished.\n");

    // Copy Result Device -> Host
    printf("Copying result from GPU...\n");
    cudaMemcpy(h_output_gpu, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Copying done.\n");


    // --- Cleanup cuDNN
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    if (workspace_bytes > 0)
        cudaFree(d_workspace);
    cudnnDestroy(cudnn);


    if (CPU_verify)
    {
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