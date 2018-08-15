#!/bin/bash
#Copyright (c) 2018 ETH Zurich, Lukas Cavigelli

#Titan X: compute_52, TX1: compute_53, GTX1080Ti: compute_61, TX2: compute_62
nvccflags="-O3 --use_fast_math -std=c++11 -Xcompiler '-fopenmp' --shared --gpu-architecture=compute_52 --compiler-options -fPIC --linker-options --no-undefined"
nvcc -o cbconv2d_cg_backend_$(uname -i).so cbconv2d_cg_backend.cu $nvccflags
nvcc -o cbconv2d_fg_backend_$(uname -i).so cbconv2d_fg_backend.cu $nvccflags

nvccflags="-O3 --use_fast_math -std=c++11 -Xcompiler '-fopenmp' --shared --gpu-architecture=compute_61 --compiler-options -fPIC --linker-options --no-undefined"
nvcc -o cbconv2d_cg_half_backend_$(uname -i).so cbconv2d_cg_half_backend.cu $nvccflags
