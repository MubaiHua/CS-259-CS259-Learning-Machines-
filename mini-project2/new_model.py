import math
import pandas as pd
import numpy as np

# RTX 5070 Data from Estimate and NCU Perf metrics
MEMORY_BANDWIDTH_Bps = 672e9 * 0.0252
L2_TO_L1_BANDWIDTH_Bps =3.5e12 * 0.1109
SINGLE_PRECISION_PERFORMANCE_FLOPs = 30.87e12 

DTYPE_SIZE_BYTES = 4 # FP32

def estimate_conv_tiled_performance_refined(
    B=1, Ni=64, Nx=32, Ny=32, Nn=128, Kx=3, Ky=3,
    TILE_OY_K=8, TILE_OX_K=8, TILE_NN_K=8, TILE_NI_K=8,
    # CUDA Block Dimensions (from launch configuration in conv2d_main.cu)
    BLOCK_DIM_X=8, BLOCK_DIM_Y=8
):


    # --- 1. Derived Problem Constants ---
    Ox = Nx - Kx + 1
    Oy = Ny - Ky + 1

    # Each MAC is 2 FLOPs.
    total_flops = 2.0 * B * Nn * Oy * Ox * Ni * Ky * Kx
    T_compute = total_flops / SINGLE_PRECISION_PERFORMANCE_FLOPs

    # Number of filter-blocks (TILE_NN_K filters per block) per batch-image
    num_fblk_per_batch = math.ceil(Nn / TILE_NN_K)
    # Grid dimensions based on output feature map tiling by block dimensions
    grid_dim_x = math.ceil(Ox / BLOCK_DIM_X)
    grid_dim_y = math.ceil(Oy / BLOCK_DIM_Y)
    grid_dim_z = B * num_fblk_per_batch
    total_blocks_launched = grid_dim_x * grid_dim_y * grid_dim_z
    
    if total_blocks_launched == 0: # Should not happen if Ox, Oy > 0
        total_blocks_launched = 1


    # Based on unique data that must be read/written from/to DRAM over the whole kernel.
    # This assumes perfect caching above DRAM  for any inter-block data reuse.
    bytes_input_unique = B * Ni * Ny * Nx * DTYPE_SIZE_BYTES
    bytes_weights_unique = Nn * Ni * Ky * Kx * DTYPE_SIZE_BYTES
    bytes_output_unique = B * Nn * Oy * Ox * DTYPE_SIZE_BYTES
    total_dram_io_bytes = bytes_input_unique + bytes_weights_unique + bytes_output_unique
    T_dram_io = total_dram_io_bytes / MEMORY_BANDWIDTH_Bps

    # This models the total data loaded into shared memory by all blocks from L2/global memory.

    # Size of input patch defined by TILE_OY_K, TILE_OX_K. One such patch per Ni.
    input_patch_elements_for_smem_in = (TILE_OY_K + Ky - 1) * (TILE_OX_K + Kx - 1)
    bytes_input_to_sm_per_block = Ni * input_patch_elements_for_smem_in * DTYPE_SIZE_BYTES

    # The weight tile (TILE_NN_K * TILE_NI_K * Kx * Ky elements) is loaded
    # ceil(Ni / TILE_NI_K) * TILE_NI_K times (approx Ni times) per block due to the
    # current kernel loop structure.
    elements_in_one_smem_w_load = TILE_NN_K * TILE_NI_K * Ky * Kx

    effective_num_inner_ni_loops = 0
    for ni0_loop_iter in range(math.ceil(Ni / TILE_NI_K)):
        start_ni = ni0_loop_iter * TILE_NI_K
        count_in_this_iteration = 0
        for ni_inner_loop_iter in range(TILE_NI_K):
            if (start_ni + ni_inner_loop_iter) < Ni:
                count_in_this_iteration +=1
        effective_num_inner_ni_loops += count_in_this_iteration
    
    # Each of these 'effective_num_inner_ni_loops' iterations loads the W_TILE_SIZE.
    bytes_weights_to_sm_per_block = effective_num_inner_ni_loops * elements_in_one_smem_w_load * DTYPE_SIZE_BYTES


    # Total bytes loaded into shared memory by ALL blocks
    total_bytes_loaded_into_all_sm = total_blocks_launched * (bytes_input_to_sm_per_block + bytes_weights_to_sm_per_block)
    T_l2_sm_io = total_bytes_loaded_into_all_sm / L2_TO_L1_BANDWIDTH_Bps
    
    # This estimates pressure on shared memory read bandwidth.
    # Data read from smem_in by ONE block's threads:
    # Each thread (BLOCK_DIM_X * BLOCK_DIM_Y) performs TILE_NN_K * Ni * Ky * Kx reads for inputs.
    # bytes_sm_input_to_reg_per_block = BLOCK_DIM_X * BLOCK_DIM_Y * TILE_NN_K * Ni * Ky * Kx * DTYPE_SIZE_BYTES
    # Data read from smem_w by ONE block's threads:
    # bytes_sm_weights_to_reg_per_block = BLOCK_DIM_X * BLOCK_DIM_Y * TILE_NN_K * Ni * Ky * Kx * DTYPE_SIZE_BYTES
    

    bottleneck_time = max(T_compute, T_dram_io, T_l2_sm_io)
    T_total = bottleneck_time

    df = pd.DataFrame({
        "Component": [
            "Compute Bound Time",
            "DRAM I/O Bound Time (Unique Data)",
            "L2-to-SM I/O Bound Time (All SM Loads)"
        ],
        "Time (s)": [
            T_compute,
            T_dram_io,
            T_l2_sm_io
        ],
        "Metric (FLOPs or Bytes)": [
            total_flops,
            total_dram_io_bytes,
            total_bytes_loaded_into_all_sm,
        ],
         "Bottleneck Candidate Time (s)": [
            T_compute,
            T_dram_io,
            T_l2_sm_io
        ]
    })
    df.loc[len(df.index)] = ['Overall Estimated Time (Max + Startup)', T_total, total_flops if bottleneck_time == T_compute else (total_dram_io_bytes if bottleneck_time == T_dram_io else total_bytes_loaded_into_all_sm) , T_total]


    return df, T_total

if __name__ == '__main__':
    B, Ni, Nx, Ny, Nn, Kx, Ky = 16, 64, 224, 224, 64, 3, 3
    
    TILE_OY_K, TILE_OX_K, TILE_NN_K, TILE_NI_K = 8,8,8,8
    
    BLOCK_DIM_X, BLOCK_DIM_Y = 8, 8

    df_refined, T_total_refined = estimate_conv_tiled_performance_refined(
        B=B, Ni=Ni, Nx=Nx, Ny=Ny, Nn=Nn, Kx=Kx, Ky=Ky,
        TILE_OY_K=TILE_OY_K, TILE_OX_K=TILE_OX_K,
        TILE_NN_K=TILE_NN_K, TILE_NI_K=TILE_NI_K,
        BLOCK_DIM_X=BLOCK_DIM_X, BLOCK_DIM_Y=BLOCK_DIM_Y
    )

    print("\nRefined Model Results:")

    df_refined_display = df_refined.copy()
    if "Time (s)" in df_refined_display.columns:
        df_refined_display["Time (ms)"] = df_refined_display["Time (s)"] * 1000
    if "Bottleneck Candidate Time (s)" in df_refined_display.columns:
        df_refined_display["Bottleneck Candidate Time (ms)"] = df_refined_display["Bottleneck Candidate Time (s)"] * 1000
    
    pd.options.display.float_format = '{:.6g}'.format
    print(df_refined_display[['Component', 'Time (ms)', 'Metric (FLOPs or Bytes)', 'Bottleneck Candidate Time (ms)']].to_string(index=False))
    print(f"\nOverall estimated time: {T_total_refined * 1e3:.3f} ms")