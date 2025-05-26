We provide parallel CUDA implementations of eleven (existing and new) HSI gradient approaches for GPU execution. The `main.cu` file illustrates the execution of the algorithm **MorphL1** applied on **Indian Pines** dataset included in the **data** directory.

We keep the implementations of the algorithms independent to easily execute the code of an algorithm without introducing dependencies.

The table below provides the filenames and the corresponding notations mentioned in the paper:

| Filename (.cu)        | Notation      | Description                                                                                                                                                |
|-----------------------|---------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| gm_gray_sum           | GGSum         | Sum of the grayscale gradients of all bands                                                                                                                |
| gm_frobenius          | GGEuc         | Euclidean (Frobenius / L<sub>2</sub>) norm of the grayscale gradients of all bands                                                                         |
| gm_di_zenzo           | DZ            | Di Zenzo's gradient magnitude calculation approach                                                                                                         |
| gm_sapiro             | Sa            | Sapiro's gradient magnitude calculation approach                                                                                                           |
| gm_morph_max          | MGMax         | Maximum of morphological gradients of all bands                                                                                                            |
| gm_morph_sum          | MGSum         | Sum of morphological gradients of all bands                                                                                                                |
| gm_morph_euc          | MGEuc         | Euclidean (L<sub>2</sub>) norm of morphological gradients of all bands                                                                                     |
| gm_morph_abs_area_sq  | CMG           | Evans's gradient magnitude calculation approach: Greatest L<sub>2</sub> norm of intensity difference between any two pixels within the structuring element |
| **gm_morph_area**     | **MorphArea** | **Difference of the maximum and minimum areas under the curve of two pixels' spectral signatures within the structuring element**                          |
| **gm_morph_abs_area** | **MorphL1**   | **Greatest L<sub>1</sub> norm of intensity difference between any two pixels within the structuring element**                                              |
| **gm_morph_area_sq**  | **MorphSS**   | **Greatest sum of signed squared intensity differences between any two pixels within the structuring element**                                             |

The project is tested with the setup mentioned below. The code may work with lower versions of the dependencies.
## Testing Setup
- **Operating System**: Ubuntu 22.04.5 LTS
- **Processor**: 13th Gen Intel® Core™ i7-13700F × 24
- **GPU**: NVIDIA GeForce RTX 4080
- **OpenCV**: Version 4.5.4
- **CUDA**: Version 12.4
- **CMake**: Version 3.22.1

## Dependencies
- **OpenCV** >= 4.5.4
- **CUDA** >= 12.4
- **CMake** >= 3.22.1

## To run the `main.cu` file

Clone the project repository

```bash
  git clone https://github.com/HSI-Analysis/hsi-gradient-cuda.git
```

Go to the project directory

```bash
  cd hsi-gradient-cuda
```

Create and go to ```build_dir``` directory

```
  mkdir build_dir
  cd build_dir
```

Set your GPU compute capability in `set_property(TARGET gm_cuda PROPERTY CUDA_ARCHITECTURES <your-gpu-compute-capability-here>)` in the file `CMakeLists.txt`.

Compile and run the code

```bash
cmake ..
make
./gm_cuda
```
