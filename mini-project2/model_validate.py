import subprocess
import re
import pandas as pd
import os

CONV2D_MAIN_EXECUTABLE = "./conv2d_main"

KX = 3
KY = 3

DTYPE_SIZE = 4

# PEAK_FLOPS = 14.9e12
# BW_L1 = 1000e9
# BW_L2 = 900e9
# BW_DRAM = 652.8e9

REUSE_INPUT_L1 = 9
REUSE_WEIGHT_L1 = 128
REUSE_OUTPUT_L1 = 1
REUSE_INPUT_L2 = 2
REUSE_WEIGHT_L2 = 10
REUSE_OUTPUT_L2 = 1

TILE_SIZE = 8

# nx_values = [32, 64, 128, 224]
# ny_values = [32, 64, 128, 224]
# ni_values = [16, 32, 64]
# nn_values = [16, 32, 64]
ni_nn_pairs = [(128, 128), (256, 256)]
nx_ny_pairs = [(128, 128), (256, 256)]
b_values = [8, 16]

from new_model import estimate_conv_tiled_performance_refined
from conv2d_perf_model import estimate_conv_kernel_bottleneck

def run_experiment(nx, ny, ni, nn, b):
    """
    Runs the conv2d_main executable and extracts the GPU kernel time.
    """
    command = [
        CONV2D_MAIN_EXECUTABLE,
        str(nx),
        str(ny),
        str(ni),
        str(nn),
        str(b),
        "False",  # CPU_verify = False
    ]
    try:
        print(f"Running: {' '.join(command)}")
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, timeout=300
        )  # 5 min timeout
        output_lines = result.stdout.splitlines()

        gpu_time_ms = None
        for line in output_lines:
            # Regex to find "GPU Kernel finished in X.XXX ms."
            match = re.search(r"GPU Kernel finished in\s*([0-9]+\.?[0-9]*)\s*ms", line)
            if match:
                gpu_time_ms = float(match.group(1))
                print(f"  Extracted GPU time: {gpu_time_ms:.3f} ms")
                break

        if gpu_time_ms is None:
            print(
                f"  WARNING: Could not extract GPU time for params: Nx={nx}, Ny={ny}, Ni={ni}, Nn={nn}, B={b}"
            )
            print("  Full output:")
            for line in output_lines:
                print(f"    {line}")
        return gpu_time_ms

    except subprocess.CalledProcessError as e:
        print(
            f"ERROR: conv2d_main failed for params: Nx={nx}, Ny={ny}, Ni={ni}, Nn={nn}, B={b}"
        )
        print(f"  Return code: {e.returncode}")
        print(f"  Stdout: {e.stdout}")
        print(f"  Stderr: {e.stderr}")
        return None
    except subprocess.TimeoutExpired:
        print(
            f"ERROR: conv2d_main timed out for params: Nx={nx}, Ny={ny}, Ni={ni}, Nn={nn}, B={b}"
        )
        return None
    except FileNotFoundError:
        print(f"ERROR: '{CONV2D_MAIN_EXECUTABLE}' not found. Please compile it first.")
        exit(1)


def main():
    """
    Main function to run experiments and generate the results table.
    """
    results_data = []

    if not os.path.exists(CONV2D_MAIN_EXECUTABLE):
        print(f"Error: Executable '{CONV2D_MAIN_EXECUTABLE}' not found.")
        print(
            f"Please compile your CUDA code (e.g., using compile_conv2d.sh) and place the executable in the current directory."
        )
        return

    print("Starting experiments...")
    for b_val in b_values:
        # for ni_val in ni_values:
        #     for nx_val in nx_values:
        #         for ny_val in ny_values:
        #             for nn_val in nn_values:
        for ni_val, nn_val in ni_nn_pairs:
            for nx_val, ny_val in nx_ny_pairs:
                print("-" * 50)
                print(
                    f"Parameters: B={b_val}, Ni={ni_val}, Nx={nx_val}, Ny={ny_val}, Nn={nn_val}, Kx={KX}, Ky={KY}"
                )

                # Run the actual CUDA program
                actual_gpu_time_ms = run_experiment(
                    nx_val, ny_val, ni_val, nn_val, b_val
                )

                # _, model_predicted_time_s = estimate_conv_tiled_performance_refined(
                #     B=b_val,
                #     Ni=ni_val,
                #     Nx=nx_val,
                #     Ny=ny_val,
                #     Nn=nn_val,
                #     Kx=KX,
                #     Ky=KY,
                #     TILE_OY_K=TILE_SIZE, 
                #     TILE_OX_K=TILE_SIZE, 
                #     TILE_NN_K=TILE_SIZE, 
                #     TILE_NI_K=TILE_SIZE,
                #     BLOCK_DIM_X=TILE_SIZE, 
                #     BLOCK_DIM_Y=TILE_SIZE
                # )

                _, model_predicted_time_s = estimate_conv_kernel_bottleneck(
                    B=b_val,
                    Ni=ni_val,
                    Nx=nx_val,
                    Ny=ny_val,
                    Nn=nn_val,
                    Kx=KX,
                    Ky=KY,
                )
                
                model_predicted_time_ms = model_predicted_time_s * 1000  # Convert to ms
                print(f"  Model predicted time: {model_predicted_time_ms:.3f} ms")

                results_data.append(
                    {
                        "B": b_val,
                        "Ni": ni_val,
                        "Nx": nx_val,
                        "Ny": ny_val,
                        "Nn": nn_val,
                        "Kx": KX,
                        "Ky": KY,
                        "Actual GPU Time (ms)": (
                            actual_gpu_time_ms
                            if actual_gpu_time_ms is not None
                            else "N/A"
                        ),
                        "Model Predicted Time (ms)": (
                            model_predicted_time_ms
                            if model_predicted_time_ms is not None
                            else "N/A"
                        ),
                    }
                )

    # Create a Pandas DataFrame for the results
    df_results = pd.DataFrame(results_data)

    print("\n" + "=" * 70)
    print("Experiment Results Summary")
    print("=" * 70)
    print(df_results.to_string())

    df_results.to_csv("conv2d_experiment_results.csv", index=False)
    print("\nResults also saved to conv2d_experiment_results.csv")


if __name__ == "__main__":
    main()
