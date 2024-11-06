# Parallel Gaussian Blur

This repository contains two parallel implementations of a **Gaussian Blur** algorithm applied to BMP images, designed to enhance performance through parallel computing techniques:
1. **OpenMP Implementation** – Executes Gaussian blur on the CPU with parallelized loops.
2. **CUDA Implementation** – Executes Gaussian blur on the GPU, leveraging CUDA's parallel processing capabilities.

## Table of Contents
- [Overview](#overview)
- [Gaussian Blur](#gaussian-blur)
- [Implementations](#implementations)
  - [OpenMP](#openmp)
  - [CUDA](#cuda)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the OpenMP Program](#running-the-openmp-program)
  - [Running the CUDA Program](#running-the-cuda-program)
- [Performance Comparison](#performance-comparison)
- [Results](#results)
- [License](#license)

## Overview

Gaussian blur is a common image processing technique that reduces image detail by smoothing pixel intensity variations. In this repository, the Gaussian blur is implemented with parallel programming for efficiency, using both **OpenMP** and **CUDA**. Each program allows users to specify the blur radius and the input BMP image size.

## Gaussian Blur

Gaussian blur works by averaging pixel values with their neighbors, where closer pixels contribute more to the average, following a Gaussian distribution. A larger radius produces a stronger blur effect.

## Implementations

### OpenMP

The OpenMP version of the Gaussian blur runs on the CPU, distributing work across multiple CPU threads to speed up processing.

### CUDA

The CUDA version uses the GPU to perform Gaussian blur operations in parallel, providing a potentially significant speedup, especially for large images and high-radius blurs.

## Installation

Clone this repository and navigate to the project directory:

```bash
git clone https://github.com/GioSed/parallel-gaussian-blur.git
cd parallel-gaussian-blur
```
Ensure you have a C compiler with OpenMP support for the OpenMP version, and an NVIDIA GPU with the CUDA toolkit installed for the CUDA version.

### Compiling the Programs
1. **OpenMP Program**:
```bash
gcc -fopenmp -o gaussian_blur_openmp.exe gaussian_blur_openmp.c -l
```
2. **CUDA Program**:

```bash
nvcc -o gaussian_blur_cuda.exe gaussian_blur_cuda.cu
```
### Usage
Both programs accept two main arguments:

1. Blur Radius – The radius of the Gaussian blur.
2. Image File – The path to the input BMP file.

### Running the OpenMP Program
To run the OpenMP program, use the following command:
```bash
gaussian_blur_openmp.exe <blur Radius> resources\<image.bmp>
```
This command applies a Gaussian blur with a specific radius to the image.bmp file in the resources folder. The program outputs:
1. Blurred image file saved with a descriptive name, e.g., image_500x500-r3-omp.bmp.
2. Execution time for the OpenMP program.

### Running the CUDA Program
To run the CUDA program, use the following command:

```bash
gaussian_blur_cuda.exe <radius> resources\<image.bmp>
```
This command applies a Gaussian blur with a specific radius of to image.bmp. The program outputs:

1. Blurred image file saved with a descriptive name, e.g., image_500x500-r3-cuda.bmp.
2. Execution time for the CUDA program.

### Performance Comparison
Both the OpenMP and CUDA versions output the execution time, allowing for a direct performance comparison. You can compare these results to the serial (non-parallelized) version of the Gaussian blur, which takes significantly longer for larger images and higher blur radii.

## Expected Performance Insights
1. OpenMP: Should show a speedup over the serial version, especially on multi-core CPUs.
2. CUDA: Generally provides the highest speedup, particularly for large images, due to the GPU's parallel processing power.

### Results
The results include:

1. Output images: Blurred BMP files for each implementation, saved with filenames indicating the radius and method used.
2. Performance metrics: Execution time for each method, which can be used to evaluate the efficiency of parallelization with OpenMP and CUDA compared to a serial implementation.

### License
This project is licensed under the MIT License - see the LICENSE file for details.
