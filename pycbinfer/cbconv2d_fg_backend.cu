//Copyright (c) 2018 ETH Zurich, Lukas Cavigelli
#include <cstdio>
#include <cmath>

extern "C" {
  
  __global__ void changeDetectionFG_kernel(const float* __restrict__ input,
                                           const float* __restrict__ prevInput,
                                           float* __restrict__ diffs, 
                                           char*  __restrict__ changeMap,
                                           const int numVals, const float threshold) {

    int valIdx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if(valIdx >= numVals)
        return;

    float d = input[valIdx] - prevInput[valIdx];
    bool pred = fabs(d) > threshold;
    changeMap[valIdx] = pred;
    if(pred)
      diffs[valIdx] = d;
  }

  void changeDetectionFG(const float* input,
                         const float* prevInput,
                         float* diffs, 
                         char* changeMap,
                         const int numVals, const float threshold) {
                  
    const int blockSize = 128;
    dim3 grid((numVals - 1)/blockSize + 1);
    dim3 block(blockSize);
    changeDetectionFG_kernel<<<grid, block>>>(input, prevInput, diffs, changeMap, numVals, threshold);
  }
  
  __global__ void updateOutputFG_kernel(const float* __restrict__ diffs,
                                      const float* __restrict__ weight,
                                      float* __restrict__ output, 
                                      const long* __restrict__ changeCoords,
                                      const int numOut, const int numIn, const int height, const int width, 
                                      const int kH, const int kW, const int numChanges) {

    int chngIdx = blockIdx.x*blockDim.x + threadIdx.x;
    if(chngIdx >= numChanges)
        return;
    
    int pos = (int) changeCoords[chngIdx];
    int ci = (int) pos/(height*width);
    int y  = (int) (pos / width) % height;
    int x  = (int) pos % width;
    
    float d = diffs[(ci*height + y)*width + x];
    
    for(int co = 0; co < numOut; co++) {
        for(int iky = 0; iky < kH; iky++) {
            for(int ikx = 0; ikx < kW; ikx++) {
                int ytot = y - iky + kH/2, xtot = x - ikx + kW/2;
                float wght = weight[((co*numIn + ci)*kH + iky)*kW + ikx];
                if(0 <= ytot && ytot < height && 0 <= xtot && xtot < width){ 
                  atomicAdd(output + (co*height + ytot)*width + xtot, wght*d);
                }
            }
        }
    }
  }

  void updateOutputFG(int gridz, int gridy, int gridx, int blockz, int blocky, int blockx,
                  const float* diffs, const float* weight, float* output, const long* changeCoords,
                  const int numOut, const int numIn, const int height, const int width, 
                  const int kH, const int kW, const int numChanges) {
                  
    dim3 grid(gridx, gridy, gridz);
    dim3 block(blockx, blocky, blockz);
    
    updateOutputFG_kernel<<<grid, block>>>(diffs, weight, output, changeCoords,
                                      numOut, numIn, height, width, 
                                      kH, kW, numChanges);
  }

  void conv2d_fg_cpu(const float* __restrict__ input, const float* __restrict__ prevInput, float* __restrict__ output, 
					 const float* __restrict__ weight, const float threshold, 
					 const int no, const int ni, const int h, const int w, const int kh, const int kw) {
				 
    const int khhalf = kh/2, kwhalf = kw/2;
#pragma omp parallel 
#pragma omp for
    for(int ci = 0; ci < ni; ci++) { 
      for(int y = 0; y < h; y++) {
        for(int x = 0; x < w; x++) {
          int iidx = (ci*h + y)*w + x;
          float diff = input[iidx] - prevInput[iidx];
          if(fabs(diff) < threshold)
            continue;
          for(int co = 0; co < no; co++) {
            for(int iky = 0; iky < kh; iky++){
              int oy = y - iky + khhalf;
              if(oy < 0 || oy >= h)
                continue;
              for(int ikx = 0; ikx < kw; ikx++){
                int ox = x - ikx + kwhalf;
                if(ox < 0 || ox >= w)
                  continue;
                int oidx = (co*h + oy)*w + ox;
                output[oidx] += diff * weight[((co*ni + ci)*kh + iky)*kw + ikx];
              }
            }
          }
        }
      }
    }
  }
}
    
