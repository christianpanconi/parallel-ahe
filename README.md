# Parallel Image Histogram Equalization

Parallel Adaptive Histogram Equalization using **OpenMP** and **CUDA**.\
The project purpose is comparing different parallelization techniques for the [*Adaptive Histogram Equalization*](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization) algorithm (without interpolation).

## Dependencies
These dependecies are **required** to build the project:
* [libjpeg-turbo](https://libjpeg-turbo.org/)
* [Google benchmark](https://github.com/google/benchmark)

## Build
Clone this repository in a `<project_root>` folder.
Then:
```bash
mkdir <project_root>/build
cd <project_root>/build
cmake -DCMAKE_BUILD_TYPE:STRING=Release ..
make all
```

The two executables can be found inside the `<project_root>/build/bin` folder.

## Running equalization and benchmarks
#### hist-equalization
The `hist-equaliztion` executable can be used to perform the equalization on a single image using either a sequential or a parallel implementation (OpenMP or CUDA).\
It supports JPEG and uncompressed images as input but only uncompressed output (for exa).\
Basic usages:
```bash
# Sequential
hist-equalization -f input_img.ppm -ws 63 -o equalized_img.ppm

# OpenMP parallel (-t option followed by the desired number of threads)
hist-equalization -t 16 -f input_img.ppm -ws 63 -o equalized_img.ppm

# CUDA parallel (--gpu), bi-directional algorithm (-e bi)
hist-equalization --gpu -e bi -f input_img.ppm -ws 63 -o equalized_img.ppm
```
The core arguments are `-f` to specify the input image, `-ws` to specify the window size and `-e mono|bi` to choose between the mono-directional or the bi-directional version.\
Use `hist-equalization -h` for more informations on the launch arguments.

#### benchmark-hist-equalization
The `benchmark-hist-equalization` executable can be used to run benchmarks for the various implementations.\
It uses the *Google benchmark* library to organize and launch the tests, as well as reporting execution times and some few useful metrics.
The Google benchmark command line arguments are supported, along with the equalization arguments (same `hist-equlization` args).\
Basic usage:
```bash
benchmark-hist-equalization -f input_img.ppm -ws 63
benchmark-hist-equalization -t 16 -f input_img.ppm -ws 63
benchmark-hist-equalization --gpu -e bi -f input_img.ppm -ws 63
```

The number of iterations for each benchmark is determined automatically by the Google benchmark library. To specify a fixed number of iterations per benchmark use the `-bmits` option followed by the desired number of iterations like:
```bash
benchmark-hist-equalization --gpu -e bi -f input_img.ppm -ws 63 -bmits 10
```

The `-mws` option can be specified to run multiple benchmarks on the same image varying the window size:
```bash
benchmark-hist-equalization --gpu -e bi -f input_img.ppm -bmits 10 -mws 63,125,255
```

Finally an example usage including some Google benchmark options:
```bash
benchmark-hist-equalization --gpu -e bi -f input_img.ppm -ws 63 -bmits 10 \
  --benchmark_repetitions=10 \
  --benchmark_out_format=csv \
  --benchmark_out=bmk_results.csv

```
