#include <filesystem>
#include <opencv2/opencv.hpp>

#include "gm_morph_abs_area.cu"

std::vector<cv::Mat> read_to_vector(const std::string& filename) {
    std::vector<cv::Mat> image_stack;
    if (!cv::imreadmulti(filename, image_stack, cv::IMREAD_UNCHANGED)) {
        std::cerr << "Could not read the TIFF file." << std::endl;
    }
    return image_stack;
}

std::uint16_t* flatten_rmbl(const std::vector<cv::Mat>& image_stack) {
    int height = image_stack[0].size[0];
    int width = image_stack[0].size[1];
    int channels = image_stack.size();

    std::uint16_t* hyperspectral_matrix;
    cudaMallocHost((void **)&hyperspectral_matrix, channels * height * width * sizeof(std::uint16_t));

    for (int c = 0; c < channels; ++c) {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                hyperspectral_matrix[j + width * (i + height * c)] = image_stack[c].at<std::uint16_t>(i, j);
            }
        }
    }

    return hyperspectral_matrix;
}

void normalize_image8(const cv::Mat& image, cv::Mat& dest) {
    double min, max;
    cv::minMaxLoc(image, &min, &max);
    image.convertTo(dest, CV_8U, std::numeric_limits<uint8_t>::max() / max);
}

int main() {
    std::string path_to_read = "../data/";
    std::string path_to_write = "../results/";
    std::filesystem::create_directory(path_to_write);

    std::vector<cv::Mat> image_stack = read_to_vector(path_to_read + "indian_pines.tif");
    std::uint16_t* h_image = flatten_rmbl(image_stack);

    int height = image_stack[0].size[0];
    int width = image_stack[0].size[1];
    int bands = image_stack.size();

    int* h_gm_morph_abs_area;
    cudaMallocHost((void **)&h_gm_morph_abs_area, height * width * sizeof(int));
    gm_morph_abs_area(width, height, bands, h_image, h_gm_morph_abs_area);

    cv::Mat gm_mat_morph_abs_area = cv::Mat(height, width, CV_32S, h_gm_morph_abs_area);
    normalize_image8(gm_mat_morph_abs_area, gm_mat_morph_abs_area);
    cv::imwrite(path_to_write + "gm_morph_abs_area.png", gm_mat_morph_abs_area);
    cudaFreeHost(h_gm_morph_abs_area);

    cudaFreeHost(h_image);

    return 0;
}
