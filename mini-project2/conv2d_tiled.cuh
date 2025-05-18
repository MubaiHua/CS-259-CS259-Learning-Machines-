#pragma once
/**********************************************************************
*  Header-only, tiled 2-D convolution kernel (NCHW layout)
*  Include this file in any .cu that needs it.
*********************************************************************/
#include <cuda_runtime.h>
#include <stddef.h>   // size_t

template<
    int TILE_OY,      // output-tile height (e.g.  8, 16)
    int TILE_OX,      // output-tile width  (e.g. 16, 32)
    int TILE_NN,      // #output channels (filters) a thread-block computes
    int TILE_NI,      // #input  channels each thread processes per sweep
    int Kx,  int Ky   // kernel spatial size (compile-time helps unroll)
>

__global__ void conv2d_tiled(float *__restrict__ output,
                             const float *__restrict__ input,
                             const float *__restrict__ weights,
                             int B, int Nx, int Ny,
                             int Ni, int Nn)
{
    /* ----------- derived constants ----------- */
    const int Ox = Nx - Kx + 1;
    const int Oy = Ny - Ky + 1;

    /* ----------- NEW: correct z-axis mapping ----------- */
    /* how many filter-blocks (each handles TILE_NN filters) per batch‐image */
    const int NUM_FBLK = (Nn + TILE_NN - 1) / TILE_NN;

    /* batch index and first filter handled by this thread-block */
    const int b      = blockIdx.z / NUM_FBLK;
    const int blk_nn = (blockIdx.z % NUM_FBLK) * TILE_NN;

    /* x / y inside the feature map – unchanged */
    const int blk_ox = blockIdx.x * TILE_OX;
    const int blk_oy = blockIdx.y * TILE_OY;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;


    const int IN_TILE_SIZE  = (TILE_OY + Ky - 1) * (TILE_OX + Kx - 1);
    const int W_TILE_SIZE   = TILE_NN * TILE_NI * Ky * Kx;
    extern __shared__ float smem[];        // dynamically sized
    float* smem_in  = smem;                    // first page: input patch
    float* smem_w   = smem + IN_TILE_SIZE;     // second page: weights

    const int SM_W = TILE_OX + Kx - 1;     // stride in shared memory

    /* -------- guard: block may be partially outside image -------- */
    if (blk_ox >= Ox || blk_oy >= Oy || b >= B) return;

    /* Loop over Ni in chunks of TILE_NI ---------------------------- */
    float accum[TILE_NN] = {0.0f};         // per-thread registers

    for (int ni0 = 0; ni0 < Ni; ni0 += TILE_NI)          // sweep input-channel blocks
    {
        /* loop inside the block -------------------------------------------- */
        for (int ni = 0; ni < TILE_NI && (ni0 + ni) < Ni; ++ni)
        {
            const int gin = ni0 + ni;                    // global input-channel idx

            /* 1. cooperative load of *this* channel’s patch into shared memory */
            for (int y = ty; y < TILE_OY + Ky - 1; y += blockDim.y)
                for (int x = tx; x < TILE_OX + Kx - 1; x += blockDim.x)
                {
                    int iy = blk_oy + y;
                    int ix = blk_ox + x;
                    if (iy < Ny && ix < Nx) {
                        size_t in_idx = ((size_t)b   * Ni * Ny * Nx) +
                                        ((size_t)gin * Ny * Nx)      +
                                        ((size_t)iy  * Nx) + ix;
                        smem_in[y * SM_W + x] = input[in_idx];
                    }
                }
            __syncthreads();

            /* ----- load TILE_NN × TILE_NI × Kx × Ky weights into smem_w ----- */
            int tid   = ty * blockDim.x + tx;                  // 0‥(TB-1)
            int stride = blockDim.x * blockDim.y;

            for (int idx = tid; idx < W_TILE_SIZE; idx += stride)
            {
                int tmp = idx;
                int kx  =  tmp % Kx;  tmp /= Kx;
                int ky  =  tmp % Ky;  tmp /= Ky;
                int ni  =  tmp % TILE_NI; tmp /= TILE_NI;
                int dn  =  tmp;                   // 0‥TILE_NN-1

                int g_ni = ni0 +  ni;             // global input-channel
                int g_nn = blk_nn + dn;           // global output-channel

                if (g_ni < Ni && g_nn < Nn) {
                    size_t w_idx = ((size_t)g_nn * Ni * Ky * Kx) +
                                ((size_t)g_ni * Ky * Kx)      +
                                ky * Kx + kx;
                    smem_w[idx] = weights[w_idx];
                } else {
                    smem_w[idx] = 0.0f;           // pad for edges & last tiles
                }
            }
            __syncthreads();                       // weights ready

            /* 2. do the 3×3 (or Kx×Ky) MAC for this channel */
            int ox = blk_ox + tx;
            int oy = blk_oy + ty;
            if (ox < Ox && oy < Oy) {
                #pragma unroll
                for (int ky = 0; ky < Ky; ++ky)
                #pragma unroll
                for (int kx = 0; kx < Kx; ++kx)
                {
                    float v = smem_in[(ty + ky) * SM_W + (tx + kx)];

                    // size_t w_base = (size_t)(gin * Ky * Kx) + ky * Kx + kx;
                    /* add contribution from each of the TILE_NN filters */
                    #pragma unroll
                    for (int dn = 0; dn < TILE_NN; ++dn) {
                        int nn = blk_nn + dn;
                        if (nn < Nn) {
                            int w_off = (((dn * TILE_NI + ni) * Ky) + ky) * Kx + kx;
                            float w = smem_w[w_off];
                            accum[dn] += v * w;
                        }
                    }
                }
            }
            __syncthreads();            // reuse smem for next channel
        }
    }

    /* ===== 3. write back ===== */
    if (blk_ox + tx < Ox && blk_oy + ty < Oy)
    {
        #pragma unroll
        for (int dn = 0; dn < TILE_NN; ++dn)
        {
            const int nn = blk_nn + dn;
            if (nn < Nn)
            {
                const size_t out_idx = ((size_t)b  * Nn * Oy * Ox) +
                                        (size_t)nn * Oy * Ox +
                                        (size_t)(blk_oy + ty) * Ox +
                                        (blk_ox + tx);
                output[out_idx] = accum[dn];
            }
        }
    }
}