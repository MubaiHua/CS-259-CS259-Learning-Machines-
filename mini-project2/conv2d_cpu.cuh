#pragma once

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