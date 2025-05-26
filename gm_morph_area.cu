#include <cstdint>

#define BLOCK_X_2D 16
#define BLOCK_Y_2D 16

__global__ void find_area_kernel(int NX, int NY, int NC, std::uint16_t* d_image, int* d_area) {
    int i, j, ind;

    // define global indices
    i = threadIdx.x + blockIdx.x * blockDim.x;
    j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < NX && j < NY) { // && i >= 0 && j >= 0
        ind = i + j * NX;
        int sum = 0;
        for (int c = 0; c < NC; ++c) {
            sum += d_image[ind + c * NX * NY];
        }
        d_area[ind] = sum;
    }
}

__global__ void gm_morph_area_kernel(int NX, int NY, int* d_area, int* d_gm) {
    int i, j, ind;
    int ni, nj, nind;
    int max_area, min_area;

    int off_x[] = {-1, 0, 1, -1, 1, -1, 0, 1};
    int off_y[] = {-1, -1, -1, 0, 0, 1, 1, 1};

    // define global indices
    i = threadIdx.x + blockIdx.x * blockDim.x;
    j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < NX && j < NY) { // && i >= 0 && j >= 0
        ind = i + j * NX;
        max_area = min_area = d_area[ind];

        for (int k = 0; k < 8; k++) {
            ni = i + off_x[k];
            nj = j + off_y[k];

            if (ni >= 0 && ni < NX && nj >= 0 && nj < NY) {
                nind = ni + nj * NX;
                if (d_area[nind] > max_area) {
                    max_area = d_area[nind];
                } else if (d_area[nind] < min_area) {
                    min_area = d_area[nind];
                }
            }
        }
        d_gm[ind] = max_area - min_area;
    }
}


void gm_morph_area(int NX, int NY, int NC, std::uint16_t* h_image, int* h_gm) {
    // initialisations
    int bx, by;
    std::uint16_t* d_image;
    int* d_area;
    int* d_gm;

    // allocate memory for arrays
    cudaMalloc((void**) &d_image, NX * NY * NC * sizeof(std::uint16_t));
    cudaMalloc((void**) &d_area, NX * NY * sizeof(int));
    cudaMalloc((void**) &d_gm, NX * NY * sizeof(int));

    // data transfer CPU to GPU and initialisation
    cudaMemcpy(d_image, h_image, NX * NY * NC * sizeof(std::uint16_t), cudaMemcpyHostToDevice);

    // GPU processing
    bx = 1 + (NX - 1) / BLOCK_X_2D;
    by = 1 + (NY - 1) / BLOCK_Y_2D;

    dim3 dimGrid(bx, by);
    dim3 dimBlock(BLOCK_X_2D, BLOCK_Y_2D);

    find_area_kernel<<<dimGrid, dimBlock>>>(NX, NY, NC, d_image, d_area);
    gm_morph_area_kernel<<<dimGrid, dimBlock>>>(NX, NY, d_area, d_gm);

    // data transfer GPU to CPU
    cudaMemcpy(h_gm, d_gm, NX * NY * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_image);
    cudaFree(d_area);
    cudaFree(d_gm);
}

