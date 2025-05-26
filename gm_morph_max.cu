#include <cstdint>

#define BLOCK_X_2D 16
#define BLOCK_Y_2D 16

__global__ void gm_morph_max_kernel(int NX, int NY, int NC, std::uint16_t* d_image, std::uint16_t* d_gm) {
    int i, j, ind;
    int ni, nj, nind;
    std::uint16_t max_val, min_val, nval;
    std::uint16_t max_marginal = 0;

    int off_x[] = {-1, 0, 1, -1, 1, -1, 0, 1};
    int off_y[] = {-1, -1, -1, 0, 0, 1, 1, 1};

    // define global indices
    i = threadIdx.x + blockIdx.x * blockDim.x;  
    j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < NX && j < NY) { // && i >= 0 && j >= 0
        ind = i + j * NX;
        for (int c = 0; c < NC; c++) {
            max_val = min_val = d_image[ind + c * NX * NY];
            for (int k = 0; k < 8; k++) {
                ni = i + off_x[k];
                nj = j + off_y[k];

                if (ni >= 0 && ni < NX && nj >= 0 && nj < NY) {
                    nind = ni + nj * NX;
                    nval = d_image[nind + c * NX * NY];
                    if (nval > max_val) {
                        max_val = nval;
                    } else if (nval < min_val) {
                        min_val = nval;
                    }
                }
            }
            if (max_val - min_val > max_marginal) {
                max_marginal = max_val - min_val;
            }
        }
        d_gm[ind] = max_marginal;
    }
}


void gm_morph_max(int NX, int NY, int NC, std::uint16_t* h_image, std::uint16_t* h_gm) {
    // initialisations
    int bx, by;
    std::uint16_t* d_image;
    std::uint16_t* d_gm;

    // allocate memory for arrays
    cudaMalloc((void**) &d_image, NX * NY * NC * sizeof(std::uint16_t));
    cudaMalloc((void**) &d_gm, NX * NY * sizeof(std::uint16_t));

    // data transfer CPU to GPU and initialisation
    cudaMemcpy(d_image, h_image, NX * NY * NC * sizeof(std::uint16_t), cudaMemcpyHostToDevice);

    // GPU processing
    bx = 1 + (NX - 1) / BLOCK_X_2D;
    by = 1 + (NY - 1) / BLOCK_Y_2D;

    dim3 dimGrid(bx, by);
    dim3 dimBlock(BLOCK_X_2D, BLOCK_Y_2D);

    gm_morph_max_kernel<<<dimGrid, dimBlock>>>(NX, NY, NC, d_image, d_gm);

    // data transfer GPU to CPU
    cudaMemcpy(h_gm, d_gm, NX * NY * sizeof(std::uint16_t), cudaMemcpyDeviceToHost);
    cudaFree(d_image);
    cudaFree(d_gm);
}

