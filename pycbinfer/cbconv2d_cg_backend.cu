//Copyright (c) 2018 ETH Zurich, Lukas Cavigelli
#include <cmath>

extern "C" {

   __global__ void changeDetection_1x1_kernel(
                                         const float* __restrict__ input,
                                         float* __restrict__ inputState,
                                         bool* __restrict__ changeMap,
                                         const int width, const int height, const int nInputPlane,
                                         const float diffThreshold,
                                         const bool updateInputState) {

    // compute pixel index
    int pxlInpIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(pxlInpIdx >= height * width) return;

    //check for changes at the pixel location; any feature map can trigger a change
    bool change = false;
    for (int i = 0; i < nInputPlane; ++i) {
      int idx = i*(height*width) + pxlInpIdx; // !! this expression might need the long datatype for high res
      change |= fabs(inputState[idx] - input[idx]) > diffThreshold;
    }

    if(!change) return; // no need to mark outputs/proceed, if not changed

    // mark pixels in the support of the changed pixel for updating
    changeMap[pxlInpIdx] = true;

    //update prevInput if with copyChanges/feedback
    if(updateInputState) { // implicit (&& change), otherwise already returned
      for (int i = 0; i < nInputPlane; ++i) {
        int idx = i*(height*width) + pxlInpIdx; // !! this expression might need the long datatype for high res                                                           
        inputState[idx] = input[idx];
      }
    }
  }
  
  
  __global__ void changeDetection_kernel(
                                         const float* __restrict__ input,
                                         float* __restrict__ inputState,
                                         bool* __restrict__ changeMap,
                                         const int width, const int height, const int nInputPlane,
                                         const int kHHalf, const int kWHalf, const float diffThreshold,
                                         const bool updateInputState) {

    // compute pixel index
    int pxlInpIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(pxlInpIdx >= height * width) return;

    //check for changes at the pixel location; any feature map can trigger a change
    bool change = false;
    for (int i = 0; i < nInputPlane; ++i) {
      int idx = i*(height*width) + pxlInpIdx; // !! this expression might need the long datatype for high res
      change |= fabs(inputState[idx] - input[idx]) > diffThreshold;
    }

    if(!change) return; // no need to mark outputs/proceed, if not changed

    // mark pixels in the support of the changed pixel for updating
    int xIn = pxlInpIdx % width;
    int yIn = pxlInpIdx / width;
    for (int k = -kHHalf; k <= kHHalf; ++k) {
      int yOut = yIn + k;
      for (int l = -kWHalf; l <= kWHalf; ++l) {
        int xOut = xIn + l;
        if(yOut>=0 && yOut<height && xOut>=0 && xOut<width) {
          changeMap[yOut*width + xOut] = true;
        }
      }
    }

    //update prevInput if with copyChanges/feedback
    if(updateInputState) { // implicit (&& change), otherwise already returned
      for (int i = 0; i < nInputPlane; ++i) {
        int idx = i*(height*width) + pxlInpIdx; // !! this expression might need the long datatype for high res                                                           
        inputState[idx] = input[idx];
      }
    }
  }

  void changeDetection(int gridz, int gridy, int gridx, int blockz, int blocky, int blockx,
                       const float* __restrict__ input,
                       float* __restrict__ oldinput,
                       bool* __restrict__ changeMap,
                       const int width, const int height, const int nInputPlane,
                       const int kHHalf, const int kWHalf, const float diffThreshold,
                       const bool updateInputState) {
    dim3 grid(gridx, gridy, gridz);
    dim3 block(blockx, blocky, blockz);

    if(kHHalf == 0 and kWHalf == 0) {      
      changeDetection_1x1_kernel<<<grid, block>>>(input, oldinput, changeMap, width, height,
                                                  nInputPlane, diffThreshold, updateInputState);
    } else {
      changeDetection_kernel<<<grid, block>>>(input, oldinput, changeMap, width, height, 
                                              nInputPlane, kHHalf, kWHalf, diffThreshold, updateInputState);
    }
  }
  __global__ void changePropagation_kernel(const bool* __restrict__ changeMatrixIn,
                                           bool* __restrict__ changeMatrixOut,
                                           const int width, const int height,
                                           const int kHHalf, const int kWHalf) {

    // compute pixel index
    int pxlInpIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(pxlInpIdx >= height * width) return;

    //check if any pixel in the input range has changed
    bool change = false;
    int xOut = pxlInpIdx % width;
    int yOut = pxlInpIdx / width;
    for (int k = -kHHalf; k <= kHHalf; ++k) {
      int yIn = yOut + k;
      for (int l = -kWHalf; l <= kWHalf; ++l) {
        int xIn = xOut + l;
        if(yIn>=0 && yIn<height && xIn>=0 && xIn<width) {
          change = change || changeMatrixIn[yIn*width + xIn];
        }
      }
    }
    changeMatrixOut[yOut*width + xOut] = change;
  }

  void changePropagation(int gridz, int gridy, int gridx, int blockz, int blocky, int blockx,
                         const bool* __restrict__ changeMatrixIn,
                         bool* __restrict__ changeMatrixOut,
                         const int width, const int height,
                         const int kHHalf, const int kWHalf) {
    dim3 grid(gridx, gridy, gridz);
    dim3 block(blockx, blocky, blockz);
    
    changePropagation_kernel<<<grid, block>>>(changeMatrixIn, changeMatrixOut, 
                                              width, height, kHHalf, kWHalf);
  }

  __global__ void genXMatrix_kernel(
                                    float* columns,
                                    const float* __restrict__ input,
                                    const int* __restrict__ changeList,
                                    const int kW, const int kH,
                                    const int nInputPlane, const int width, const int height,
                                    const int numChanges) {

    const int kx = threadIdx.x;
    const int ky = threadIdx.z;
    const	int changeIdx = blockIdx.x * blockDim.y + threadIdx.y;

    if(changeIdx < numChanges) {
      int pos = changeList[changeIdx];
      int ix = pos % width + kx - (kW-1)/2;
      int iy = pos / width + ky - (kH-1)/2;
      float *dst = columns + changeIdx*(kW*kH*nInputPlane) + ky*kW+kx;

      const bool isInImage = ix>=0 && ix < width && iy>=0 && iy< height;
      for (int i = 0; i < nInputPlane; ++i) {
        dst[i*kH*kW] = isInImage ? input[(i*height + iy) * width + ix] : 0.0f;
      }
    }
  }

  void genXMatrix(int gridz, int gridy, int gridx, int blockz, int blocky, int blockx,
                  float* columns,
                  const float* __restrict__ input,
                  const int* __restrict__ changeList,
                  const int kW, const int kH,
                  const int nInputPlane, const int width, const int height,
                  const int numChanges) {
    dim3 grid(gridx, gridy, gridz);
    dim3 block(blockx, blocky, blockz);
    genXMatrix_kernel<<<grid, block>>>(columns, input, changeList, kW, kH, nInputPlane, width, height, numChanges);
  }

  __global__ void updateOutput_kernel(const float* __restrict__ columnsOut,
                                      float* output, const int* __restrict__ changeList,
                                      const int numOutputPixel, const int numChanges, const int nOutputPlane, const bool relu) {

    int count = blockIdx.x * blockDim.x + threadIdx.x;

    if(count < numChanges*nOutputPlane) {
      int outpPlane = count / numChanges;
      int changeNr =  count % numChanges;
      int pxl = changeList[changeNr];
      float v = columnsOut[count];
      v = relu && (v <= 0) ? 0 : v;
      output[outpPlane*numOutputPixel + pxl] = v;
    }
  }

  void updateOutput(int gridz, int gridy, int gridx, int blockz, int blocky, int blockx,
                    float *columnsOut, float *output, int* changeList, 
                    int numOutputPixel, int numChanges, int nOutputPlane, bool relu) {
    dim3 grid(gridx, gridy, gridz);
    dim3 block(blockx, blocky, blockz);
    updateOutput_kernel<<<grid, block>>>(columnsOut, output, changeList, numOutputPixel, numChanges, nOutputPlane, relu);
  }

  __global__ void maxPool2d_kernel(const float* __restrict__ input,
                                   float* __restrict__ output, 
                                   const int* __restrict__ changeIndexes,
                                   const int numChanges, 
                                   const int numCh, const int iheight, const int iwidth,
                                   const int oheight, const int owidth,
                                   const int stridey, const int stridex) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= numChanges)
      return;
    int pxIdx = changeIndexes[tid];
    int y = pxIdx / iwidth, x = pxIdx % iwidth;
    int yo = y / stridey, xo = x / stridex;

    for(int ch = 0; ch < numCh; ch++) {
      float v = -INFINITY;
      for(int j = 0; j < stridey; j++) {
        for(int i = 0; i < stridex; i++) {
          int yi = yo*stridey + j, xi = xo*stridex + i;
          if(yi < iheight && xi < iwidth) {
            v = max(v, input[(ch*iheight + yi)*iwidth + xi]);
          }
        }
      }
      output[(ch*oheight + yo)*owidth + xo] = v;
    }
    
  }

  void maxPool2d(int gridx, int blockx,
                 float *input, float *output, int* changeIndexes, int numChanges, 
                 int numCh, int iheight, int iwidth,
                 int oheight, int owidth,
                 int stridey, int stridex) {
                 
    //determine grid and block size based on number of changes
    dim3 grid(gridx);
    dim3 block(blockx);
    maxPool2d_kernel<<<grid, block>>>(input, output, changeIndexes, numChanges, 
                                      numCh, iheight, iwidth, oheight, owidth, stridey, stridex);
  }

}
