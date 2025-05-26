#include <cstdint>

#define BLOCK_X_2D 16
#define BLOCK_Y_2D 16

__device__ long long area_squares_difference(int limit, int step, std::uint16_t* d_image, int ind1, int ind2) {
    long long asd = 0LL, band_diff;
    int band1 = ind1, band2 = ind2;
    while (band1 < limit && band2 < limit) {
        if (d_image[band1] < d_image[band2]) {
            band_diff = d_image[band2] - d_image[band1];
            asd -= band_diff * band_diff;
        } else {
            band_diff = d_image[band1] - d_image[band2];
            asd += band_diff * band_diff;
        }
        band1 += step;
        band2 += step;
    }
    return asd;
}

__global__ void gm_morph_area_sq_kernel(int NX, int NY, int NC, std::uint16_t* d_image, long long* d_gm) {
    int i, j, ind;
    int pi, pj, pind;
    int qi, qj, qind;
    long long gm = 0LL;

    int off_x[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    int off_y[] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};

    // define global indices
    i = threadIdx.x + blockIdx.x * blockDim.x;
    j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < NX && j < NY) { // && i >= 0 && j >= 0
        ind = i + j * NX;

        for (int p = 0; p < 8; p++) {
            pi = i + off_x[p];
            pj = j + off_y[p];

            if (pi >= 0 && pi < NX && pj >= 0 && pj < NY) {
                pind = pi + pj * NX;

                for (int q = p + 1; q < 9; q++) {
                    qi = i + off_x[q];
                    qj = j + off_y[q];
                    
                    if (qi >= 0 && qi < NX && qj >= 0 && qj < NY) {
                        qind = qi + qj * NX;
                        long long asd = area_squares_difference(NX * NY * NC, NX * NY, d_image, pind, qind);
                        if (asd < 0) {
                            asd *= -1;
                        }
                        if (asd > gm) {
                            gm = asd;
                        }
                    }
                }
            }
        }
        d_gm[ind] = gm;
    }
}


void gm_morph_area_sq(int NX, int NY, int NC, std::uint16_t* h_image, long long* h_gm) {
    // initialisations
    int bx, by;
    std::uint16_t* d_image;
    long long* d_gm;

    // allocate memory for arrays
    cudaMalloc((void**) &d_image, NX * NY * NC * sizeof(std::uint16_t));
    cudaMalloc((void**) &d_gm, NX * NY * sizeof(long long));

    // data transfer CPU to GPU and initialisation
    cudaMemcpy(d_image, h_image, NX * NY * NC * sizeof(std::uint16_t), cudaMemcpyHostToDevice);

    // GPU processing
    bx = 1 + (NX - 1) / BLOCK_X_2D;
    by = 1 + (NY - 1) / BLOCK_Y_2D;

    dim3 dimGrid(bx, by);
    dim3 dimBlock(BLOCK_X_2D, BLOCK_Y_2D);

    gm_morph_area_sq_kernel<<<dimGrid, dimBlock>>>(NX, NY, NC, d_image, d_gm);

    // data transfer GPU to CPU
    cudaMemcpy(h_gm, d_gm, NX * NY * sizeof(long long), cudaMemcpyDeviceToHost);
    cudaFree(d_image);
    cudaFree(d_gm);
}

