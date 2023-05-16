# Parallel Image Histogram Equalization

Parallel Adaptive Histogram Equalization using **OpenMP** and **CUDA**.\
The project purpose is comparing different parallelization techniques for the [*Adaptive Histogram Equalization*](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization) algorithm (without interpolation).

This project includes the following:
* C++ library
* python pacakge
* test executables

## Dependencies
The C++ library has no external dependecies other than `OpenMP` and `CUDA`.

The python package requires `NumPy` to be present among the installed python packages.

The test executables depends on:
* [libjpeg-turbo](https://libjpeg-turbo.org/)
* [Google benchmark](https://github.com/google/benchmark)  

(the dependencies for the test executables can be automatically downloaded during the build process, see the Build/Install section for more informations)

## Build/Install

`CMake` version >= `3.18` is required.  
The following instructions assumes this repository cloned in a folder named `parallel-ahe`.

#### C++ library
```bash
mkdir parallel-ahe/build
cd parallel-ahe/build
cmake .. -DCMAKE_BUILD_TYPE:String=Release \
  -DCMAKE_CUDA_ARCHITECTURES=<target compute capabilities>
make all
```

The `<target compute capabilities>` requires the CC identifier, for example to compile for CC 6.1 use `-DCMAKE_CUDA_ARCHITECTURES=61`.

The kernels contained in the library will be optimized using the `__launch_bounds__` directive, the bounds are automatically determined from the specified cuda architecture.  
To disable the launch bounds optimization use the flag `-DUSE_CUDA_LAUNCH_BOUNDS=OFF`.

The test executables will be compiled along with the library and they require the dependiecies listed in the "Dependecies" section.\
Dependecies can be downloaded automatically using the option `-DDOWNLOAD_DEPENDENCIES=ON`.\
Alternatively, the tests can be skipped using `-DSKIP_TESTS=ON`.

The produced library will be located in `parallel-ahe/build/lib` and the test executables in `parallel-ahe/build/tests`.

Once the library has bees successfully built, it can be installed with:
```bash
make install
```

#### Python package

The python package can be built using the `setup.py` script:
```bash
cd parallel-ahe
python3 setup.py build --build-type=Release --cuda-archs=<target compute capabilities>
```
The `setup.py` script supports the `--disable-cuda-launch-bounds` option to disable the kernels launch bounds optimization.

Once the package has been successfully built, it can be installed using `pip`:
```bash
pip install .
```

The package is named `histogram-equalization` and can be uninstalled with:
```bash
pip uninstall histogram-equalization
```

## Running the test executables
#### hist-equalization
The `hist-equaliztion` executable can be used to perform the equalization on a single image using either a sequential or a parallel implementation (OpenMP or CUDA).\
It supports JPEG and uncompressed images as input but only uncompressed output (for example *ppm*).

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

An example usage including some Google benchmark options:
```bash
benchmark-hist-equalization --gpu -e bi -f input_img.ppm -ws 63 -bmits 10 \
  --benchmark_repetitions=10 \
  --benchmark_out_format=csv \
  --benchmark_out=bmk_results.csv

```

Use `benchmark-hist-equalization -h` for more informations on the launch arguments.
