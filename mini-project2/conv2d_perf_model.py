import numpy as np
import pandas as pd

def estimate_conv_kernel_bottleneck(
    B=1, Ni=64, Nx=32, Ny=32, Nn=128, Kx=3, Ky=3,
    dtype_size=4,                      # FP32 = 4 bytes
    peak_flops=30.87e12,               # Titan V FP32 peak
    bw_L1=1000e9,
    bw_L2=900e9,
    bw_DRAM=672e9,
    reuse_input_L1=None,
    reuse_weight_L1=None,
    reuse_output_L1=1,
    reuse_input_L2=2,
    reuse_weight_L2=10,
    reuse_output_L2=1
):
    # Output dimensions
    Ox = Nx - Kx + 1
    Oy = Ny - Ky + 1

    # Total FLOPs
    total_flops = 2 * B * Nn * Oy * Ox * Ni * Ky * Kx

    # Raw memory volumes
    input_bytes = B * Ni * Ny * Nx * dtype_size
    weight_bytes = Nn * Ni * Ky * Kx * dtype_size
    output_bytes = B * Nn * Oy * Ox * dtype_size

    # Default reuse assumptions
    if reuse_input_L1 is None:
        reuse_input_L1 = Kx * Ky
    if reuse_weight_L1 is None:
        reuse_weight_L1 = Ox * Oy

    # Traffic estimation helper
    def traffic(bytes_raw, reuse):
        return bytes_raw / reuse

    traffic_L1 = (
        traffic(input_bytes, reuse_input_L1) +
        traffic(weight_bytes, reuse_weight_L1) +
        traffic(output_bytes, reuse_output_L1)
    )
    traffic_L2 = (
        traffic(input_bytes, reuse_input_L2) +
        traffic(weight_bytes, reuse_weight_L2) +
        traffic(output_bytes, reuse_output_L2)
    )
    traffic_DRAM = input_bytes + weight_bytes + output_bytes

    # Time estimates
    T_compute = total_flops / peak_flops
    # T_L1 = traffic_L1 / bw_L1
    # T_L2 = traffic_L2 / bw_L2
    T_DRAM = traffic_DRAM / bw_DRAM
    # T_mem = max(T_L1, T_L2, T_DRAM)
    T_total = max(T_compute, T_DRAM)

    # Summary table
    # df = pd.DataFrame({
    #     "Component": ["Compute", "L1 Memory", "L2 Cache", "DRAM"],
    #     "Time (s)": [T_compute, T_L1, T_L2, T_DRAM],
    #     "Traffic (GB)": [np.nan, traffic_L1 / 1e9, traffic_L2 / 1e9, traffic_DRAM / 1e9],
    # })

    return None, T_total


df, T = estimate_conv_kernel_bottleneck(
    B=16, Ni=64, Nx=224, Ny=224, Nn=64, Kx=3, Ky=3,
    reuse_input_L1=9,
    reuse_weight_L1=128,  # 3x3 kernel; 16x8 tile size
    reuse_output_L1=1,
    reuse_input_L2=2,
    reuse_weight_L2=10,
    reuse_output_L2=1
)

print(df)
print(f"Total estimated time: {T * 1e6:.2f} µs")