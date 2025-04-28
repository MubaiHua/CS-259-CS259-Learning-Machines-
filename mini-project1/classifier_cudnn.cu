#include <cuda_runtime.h>
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h> // For fabs
#include <time.h> // For srand

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

void classifier_cpu(float *output, const float *input, const float *weights,
                    int B, int Ni, int Nn)
{
    for (int b = 0; b < B; ++b)
    {
        for (int nn = 0; nn < Nn; ++nn)
        {
            float sum = 0.0f;
            for (int ni = 0; ni < Ni; ++ni)
            {
                size_t input_idx = (size_t)b * Ni + ni;
                size_t weight_idx = (size_t)nn * Ni + ni;
                sum += input[input_idx] * weights[weight_idx];
            }
            size_t output_idx = (size_t)b * Nn + nn;
            output[output_idx] = sum;
        }
    }
}

__global__ void classifier_kernel(float *output, const float *input, const float *weights,
                                  int B, int Ni, int Nn)
{
    int nn = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;

    if (nn < Nn && b < B)
    {
        float sum = 0.0f;
        for (int ni = 0; ni < Ni; ++ni)
        {
            size_t input_idx = (size_t)b * Ni + ni;
            size_t weight_idx = (size_t)nn * Ni + ni;
            sum += input[input_idx] * weights[weight_idx];
        }
        size_t output_idx = (size_t)b * Nn + nn;
        output[output_idx] = sum;
    }
}

// --- Main Verification Function ---
int main(int argc, char *argv[])
{
    // int Ni = 25088, Nn = 4096;
    // int B = 16;

    if (argc != 5)
    {
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

    // --- Create cuDNN Handle
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // Calculate sizes
    size_t input_size = (size_t)B * Ni;
    size_t weight_size = (size_t)Nn * Ni; // Assuming Nn x Ni layout
    size_t output_size = (size_t)B * Nn;

    printf("Input: %d x %d, Weights: %d x %d, Output: %d x %d\n",
           B, Ni, Nn, Ni, B, Nn);
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
    srand(456); // Different seed than conv2d
    for (size_t i = 0; i < input_size; ++i)
        h_input[i] = (float)(rand() % 20 - 10) / 10.0f;
    // Use smaller weights for classifier, common practice
    for (size_t i = 0; i < weight_size; ++i)
        h_weights[i] = (float)(rand() % 10 - 5) / 50.0f;

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

    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    // Set input tensor: B x Ni x 1 x 1
    cudnnSetTensor4dDescriptor(input_desc,
                            CUDNN_TENSOR_NCHW,
                            CUDNN_DATA_FLOAT,
                            B, Ni, 1, 1);

    // Set filter: Nn x Ni x 1 x 1
    cudnnSetFilter4dDescriptor(filter_desc,
                            CUDNN_DATA_FLOAT,
                            CUDNN_TENSOR_NCHW,
                            Nn, Ni, 1, 1);

    // No padding, stride 1, dilation 1
    cudnnSetConvolution2dDescriptor(conv_desc,
                                    0, 0, 1, 1, 1, 1,
                                    CUDNN_CROSS_CORRELATION,
                                    CUDNN_DATA_FLOAT);

    // Set output tensor: B x Nn x 1 x 1
    cudnnSetTensor4dDescriptor(output_desc,
                            CUDNN_TENSOR_NCHW,
                            CUDNN_DATA_FLOAT,
                            B, Nn, 1, 1);


    cudnnConvolutionFwdAlgoPerf_t perf_results;
    int returnedAlgoCount = 0;

    cudnnFindConvolutionForwardAlgorithm(cudnn,
                                        input_desc,
                                        filter_desc,
                                        conv_desc,
                                        output_desc,
                                        1,
                                        &returnedAlgoCount,
                                        &perf_results);

    cudnnConvolutionFwdAlgo_t algo = perf_results.algo;
    size_t workspace_bytes = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                            input_desc,
                                            filter_desc,
                                            conv_desc,
                                            output_desc,
                                            algo,
                                            &workspace_bytes);

    void* d_workspace = nullptr;
    if (workspace_bytes > 0)
        cudaMalloc(&d_workspace, workspace_bytes);


    const float alpha = 1.0f, beta = 0.0f;

    printf("Running cuDNN Classifier...\n");

    cudnnConvolutionForward(cudnn,
                            &alpha,
                            input_desc, d_input,
                            filter_desc, d_weights,
                            conv_desc, algo,
                            d_workspace, workspace_bytes,
                            &beta,
                            output_desc, d_output);

    cudaDeviceSynchronize();
    printf("cuDNN Classifier finished.\n");

    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    if (workspace_bytes > 0)
        cudaFree(d_workspace);
    cudnnDestroy(cudnn);

        // Copy Result Device -> Host
    printf("Copying result from GPU...\n");
    cudaMemcpy(h_output_gpu, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Copying done.\n");

    if (CPU_verify)
    {
        // --- CPU Execution ---
        printf("Running Classifier CPU reference...\n");
        classifier_cpu(h_output_cpu, h_input, h_weights, B, Ni, Nn);
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