#Copyright (c) 2018 ETH Zurich, Lukas Cavigelli

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import platform
import matplotlib.pyplot as plt
#torch.random.manual_seed(10)

import cffi
ffi_fg = cffi.FFI()
ffi_fg.cdef("""
  void updateOutputFG(int gridz, int gridy, int gridx, int blockz, int blocky, int blockx,
                  const float* diffs, const float* weight, float* output, const long* changeCoords,
                  const int numOut, const int numIn, const int height, const int width, 
                  const int kH, const int kW, const int numChanges);
  
  void changeDetectionFG(const float* input,
                         const float* prevInput,
                         float* diffs, 
                         char* changeMap,
                         const int numVals, const float threshold);
  
    void conv2d_fg_cpu(const float* input, const float* prevInput, float* output, 
				 const float* weight, const float threshold, 
				 const int no, const int ni, const int h, const int w, const int kh, const int kw);
""")
package_directory_fg = os.path.dirname(os.path.abspath(__file__))
libfile_fg = os.path.join(package_directory_fg, 'cbconv2d_fg_backend_%s.so' % (platform.machine()))
C_fg = ffi_fg.dlopen(libfile_fg)

def changeDetectionFG(input, prevInput, threshold):
    diffs = input.new(input.size())
    changeMap = torch.cuda.CharTensor(input.size())
    assert(input.is_contiguous() and input.is_cuda)
    
    C_fg.changeDetectionFG(ffi_fg.cast("float *", input.data_ptr()),
                           ffi_fg.cast("float *", prevInput.contiguous().data_ptr()),
                           ffi_fg.cast("float *", diffs.contiguous().data_ptr()),
                           ffi_fg.cast("char  *", changeMap.contiguous().data_ptr()),
                           np.int32(input.numel()), 
                           np.float32(threshold))
            
    return diffs, changeMap

def updateOutputFG(diffs, weight, output, changeCoords):
    assert(output.is_contiguous())
    assert(weight.dim() == 4)
    
    numOut, numIn, kH, kW, inH, inW = weight.size(0), weight.size(1), weight.size(2), weight.size(3), output.size(-2), output.size(-1)
    numChanges = changeCoords.size(0)

    block = (128,1,1)
    grid = ((numChanges - 1)//block[0] + 1,1)
    
    C_fg.updateOutputFG(np.int32(1), np.int32(grid[1]), np.int32(grid[0]), 
                      np.int32(block[2]), np.int32(block[1]), np.int32(block[0]), 
                      ffi_fg.cast("float *", diffs.contiguous().data_ptr()),
                      ffi_fg.cast("float *", weight.contiguous().data_ptr()),
                      ffi_fg.cast("float *", output.data_ptr()),
                      ffi_fg.cast("long  *", changeCoords.contiguous().data_ptr()),
                      np.int32(numOut),
                      np.int32(numIn),
                      np.int32(inH),
                      np.int32(inW),
                      np.int32(kH),
                      np.int32(kW),
                      np.int32(numChanges))

    return output


def cbconvFG(input, prevInput, output, weight, threshold):
#    global deltaInput, changeTensor, changeCoords
    gpu = input.is_cuda
    if gpu:
        
        deltaInput, changeTensor = changeDetectionFG(input, prevInput, threshold)
        #densify / change coordinates extraction
        changeIdx = torch.nonzero(changeTensor.view(-1)) # tensor with (0, ch, y, x) coords
        if changeIdx.dim() != 0: # if any changes; otherwise dimensionality issues...
            output = updateOutputFG(deltaInput, weight, output, changeIdx)
        return output
    else:
        C_fg.conv2d_fg_cpu(ffi_fg.cast("float *", input.contiguous().data_ptr()),
                           ffi_fg.cast("float *", prevInput.contiguous().data_ptr()),
                           ffi_fg.cast("float *", output.data_ptr()),
                           ffi_fg.cast("float *", weight.contiguous().data_ptr()),
                           np.float32(threshold),
                           np.int32(weight.size(0)), np.int32(weight.size(1)), 
                           np.int32(input.size(-2)), np.int32(input.size(-1)), 
                           np.int32(weight.size(2)), np.int32(weight.size(3))
                           )
        return output

def cbconvFG_test1(gpu=True):
    filtSize = (3,3)
    threshold = 0.00
    input = torch.zeros(1,2,9,9).float().contiguous()
    if gpu:
        input = input.cuda()
    prevInput = input.clone().contiguous()
    input[0,0,1,1] = 1.0
    input[0,0,1,2] = 3.2
    input[0,0,0,5] = 1.5
    input[0,1,5:,7:].random_(0,100).add_(-50)
    prevInput[0,1,7,5] = 4.0
    weight = torch.randn(3, input.size(-3), filtSize[1], filtSize[0]).contiguous()
    if gpu:
        weight = weight.cuda() # bias missing!!!
    weight.fill_(3.0)
    #weight[0,0,0,0] = 0.125
    
    #generate stimuli and init
    outputRef = F.conv2d(F.Variable(input), F.Variable(weight), padding=tuple(s//2 for s in filtSize)).data
    prevOutput = F.conv2d(F.Variable(prevInput), F.Variable(weight), padding=tuple(s//2 for s in filtSize)).data

    output = prevOutput.clone().contiguous()
    cbconvFG(input, prevInput, output, weight, threshold)
    
    error = (output - outputRef).abs().max()
        
    #visualization for error analysis
    ic = 1
    oc = 0
    plt.subplot(3,2,1)
    plt.imshow((input)[0,ic])
    plt.title('input')
    plt.subplot(3,2,2)
    plt.imshow((prevInput)[0,ic])
    plt.title('prevInput')
    plt.subplot(3,2,3)
    plt.imshow((output)[0,oc])
    plt.title('output')
    plt.subplot(3,2,4)
    plt.imshow((prevOutput)[0,oc])
    plt.title('prevOutput')
    plt.subplot(3,2,5)
    plt.imshow((outputRef)[0,oc])
    plt.title('outputRef')
    plt.subplot(3,2,6)
    plt.imshow((output - outputRef)[0,oc])
    plt.title('outputDiff')
    
    passed = error < 1e-6
#    print(error)

    return passed


if __name__ == "__main__":   
    passed = cbconvFG_test1(gpu=False)
    if passed:
        print('cpu: ok')
    else:
        print('cpu: FAIL')
#        assert(False) 
#    passed = cbconvFG_test1(gpu=True)
#    if passed:
#        print('gpu: ok')
#    else:
#        assert(False)
    
    ffi_fg.dlclose(C_fg)

