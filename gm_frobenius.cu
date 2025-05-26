#include <cstdint>

#define BLOCK_X_2D 16
#define BLOCK_Y_2D 16

__global__ void gm_frobenius_kernel(int NX, int NY, int NC, std::uint16_t* d_image, double* d_gm) {
    int i, j, ind;
    int ni, nj, nind, nband;

    int off_x[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
    int off_y[] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
    int sobel_x[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int sobel_y[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

    double dx = 0, dy = 0;

    // define global indices
    i = threadIdx.x + blockIdx.x * blockDim.x;
    j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < NX && j < NY) { // && i >= 0 && j >= 0
        ind = i + j * NX;
        double sum_pd_sq = 0;
        // partial derivatives with Sobel kernel
        for (int c = 0; c < NC; ++c) {
            dx = 0;
            dy = 0;
            for (int k = 0; k < 9; ++k) {
                ni = i + off_x[k];
                nj = j + off_y[k];

                // boundary conditions to be consistent with BORDER_REFLECT_101 of opencv
                if (ni < 0) {
                    ni = -ni;
                }
                if (ni >= NX) {
                    ni = 2 * NX - ni - 2;
                }

                if (nj < 0) {
                    nj = -nj;
                }
                if (nj >= NY) {
                    nj = 2 * NY - nj - 2;
                }

                nind = ni + nj * NX;
                nband = nind + NX * NY * c;

                dx += sobel_x[k] * d_image[nband];
                dy += sobel_y[k] * d_image[nband];
            }
            sum_pd_sq += dx * dx + dy * dy;
        }
        d_gm[ind] = sqrt(sum_pd_sq);
    }
}

void gm_frobenius(int NX, int NY, int NC, std::uint16_t* h_image, double* h_gm) {
    // initialisations
    int bx, by;
    std::uint16_t* d_image;
    double* d_gm;

    // allocate memory for arrays
    cudaMalloc((void**) &d_image, NX * NY * NC * sizeof(std::uint16_t));
    cudaMalloc((void**) &d_gm, NX * NY * sizeof(double));

    // data transfer CPU to GPU and initialisation
    cudaMemcpy(d_image, h_image, NX * NY * NC * sizeof(std::uint16_t), cudaMemcpyHostToDevice);

    // GPU processing
    bx = 1 + (NX - 1) / BLOCK_X_2D;
    by = 1 + (NY - 1) / BLOCK_Y_2D;

    dim3 dimGrid(bx, by);
    dim3 dimBlock(BLOCK_X_2D, BLOCK_Y_2D);

    gm_frobenius_kernel<<<dimGrid, dimBlock>>>(NX, NY, NC, d_image, d_gm);

    // data transfer GPU to CPU
    cudaMemcpy(h_gm, d_gm, NX * NY * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_image);
    cudaFree(d_gm);
}
