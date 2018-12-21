#Copyright (c) 2018 ETH Zurich, Lukas Cavigelli
import os
import platform
import cffi

header = """
void changeDetection(int gridz, int gridy, int gridx, int blockz, int blocky, int blockx,
    const float* input,
    float* oldinput,
    bool* changeMatrix,
    const int width, const int height, const int nInputPlane,
    const int kHHalf, const int kWHalf, const float diffThreshold,
    const bool updateInputState);

void changePropagation(int gridz, int gridy, int gridx, int blockz, int blocky, int blockx,
                         const bool* changeMatrixIn,
                         bool* changeMatrixOut,
                         const int width, const int height,
                         const int kHHalf, const int kWHalf);

void genXMatrix(int gridz, int gridy, int gridx, int blockz, int blocky, int blockx,
    float* columns,
    const float* input,
    const int*  changeList,
    const int kW, const int kH,
    const int nInputPlane, const int width, const int height,
                const int numChanges);

void updateOutput(int gridz, int gridy, int gridx, int blockz, int blocky, int blockx,
                  float *columnsOut, float *output, int* changeList, 
                  int numOutputPixel, int numChanges, int nOutputPlane, bool relu);

void maxPool2d(int gridx, int blockx,
                 float *input, float *output, int* changeIndexes, int numChanges, 
                 int numCh, int iheight, int iwidth,
                 int oheight, int owidth,
                 int stridey, int stridex);
"""

ffi = cffi.FFI()
ffi.cdef(header)
package_directory = os.path.dirname(os.path.abspath(__file__))
lib_file = os.path.join(package_directory, 'cbconv2d_cg_backend_%s.so' % (platform.machine()))
C = ffi.dlopen(lib_file)

ffi = cffi.FFI()
ffi.cdef(header)
package_directory = os.path.dirname(os.path.abspath(__file__))
lib_file = os.path.join(package_directory, 'cbconv2d_cg_half_backend_%s.so' % (platform.machine()))
Chalf = ffi.dlopen(lib_file)

import numpy as np
import torch
import torch.nn.functional as F



def maxPool2d(input, outputState, changeIndexes, kernelSize, stride, useHalf=False):
    assert(len(kernelSize) == 2)
    assert(kernelSize == stride)
    assert(input.size(0) == 1)
    
    nc, h, w = input.size(-3), input.size(-2), input.size(-1)
    oh, ow = outputState.size(-2), outputState.size(-1)
    numChanges = len(changeIndexes)
    
    block = (64,1,1)
    grid = ((numChanges - 1)//block[0] + 1,1)
    
    sy, sx = stride
    assert(sy == 2 and sx == 2)
    
    CC = Chalf if useHalf else C
    CC.maxPool2d(np.int32(grid[0]), np.int32(block[0]), 
                          ffi.cast("float *", input.contiguous().data_ptr()),
                          ffi.cast("float *", outputState.contiguous().data_ptr()),
                          ffi.cast("int *", changeIndexes.contiguous().data_ptr()),
                          np.int32(numChanges), 
                          np.int32(nc), np.int32(h), np.int32(w), 
                          np.int32(oh), np.int32(ow),
                          np.int32(sy), np.int32(sx))
    return outputState

def genTestData():
    inC, inH, inW = 16, 400, 300#3, 15, 20
    input = torch.randn(1, inC, inH, inW).cuda()
    prevInput = input.clone()
    threshold = 0.1
    filtSize = (3,3)

    def setPoint(c,y,x,delta): prevInput[0,c,y,x] += delta
    setPoint(0, 0, 4,   1.00)
    setPoint(1, 6, 9,   0.05)
    setPoint(2,10, 4, -11.00)
    setPoint(1, 6,19,  -0.05)

    return input, prevInput, threshold, filtSize


def changeDetection(input, prevInput, filtSize, threshold, updateInputState=False, useHalf=False):
    assert(input.size() == prevInput.size() and input.dim() == 4)
#    assert(copyChanges == False)
    inC, inH, inW = input.size(-3), input.size(-2), input.size(-1)

    changeMap = torch.cuda.CharTensor(inH, inW).zero_()
    block = (128,1,1)
    grid = ((inH*inW - 1)//block[0] + 1,1)
    
    CC = Chalf if useHalf else C
    CC.changeDetection(np.int32(1), np.int32(grid[1]), np.int32(grid[0]), 
                      np.int32(block[2]), np.int32(block[1]), np.int32(block[0]), 
                      ffi.cast("float *", input.contiguous().data_ptr()),
                      ffi.cast("float *", prevInput.contiguous().data_ptr()),
                      ffi.cast("bool *", changeMap.contiguous().data_ptr()),
                      np.int32(inW),
                      np.int32(inH),
                      np.int32(inC),
                      np.int32((filtSize[0]-1)/2),
                      np.int32((filtSize[1]-1)/2),
                      np.float32(threshold),
                      ffi.cast("bool", updateInputState))
    return changeMap

def changeDetection_python(input, prevInput, filtSize, threshold):
    assert(input.size() == prevInput.size() and input.dim() == 4)
    tt = (input-prevInput).abs().ge(threshold).sum(1).gt(0)
    tt = tt.unsqueeze_(0).float()
    filt = input.new(1,1,filtSize[0],filtSize[1]).fill_(1)
    ss = F.conv2d(F.Variable(tt), F.Variable(filt),
                  padding=tuple([(sz-1)//2 for sz in filtSize]))
    res = (ss.data > 0).char()
    assert(res.dim() == 4 and res.size(0) == 1 and res.size(1) == 1)
    return res


def changeDetection_test1():
    input, prevInput, threshold, filtSize = genTestData()

    expResp = changeDetection_python(input, prevInput, filtSize, threshold)
    changeMap = changeDetection(input, prevInput, filtSize, threshold)
    passed = (expResp.char().cpu() == changeMap.cpu()).all()
    return passed


def changePropagation_python(changeMap, filtSize):
    assert(changeMap.dim() == 4 and len(filtSize) == 2)
    filtHHalf, filtWHalf = [fs//2 for fs in filtSize]
    changeMapOut = changeMap.clone()
    for ky, kx in ((ky,kx) for ky in range(filtHHalf+1) for kx in range(filtWHalf+1)):
        if ky == 0 and kx == 0:
            continue
        changeMapOut[0,0,ky:,kx:] += changeMap[0,0,:changeMap.size(2)-ky,:changeMap.size(3)-kx]
        changeMapOut[0,0,ky:,:changeMap.size(3)-kx] += changeMap[0,0,:changeMap.size(2)-ky,kx:]
        changeMapOut[0,0,:changeMap.size(2)-ky,kx:] += changeMap[0,0,ky:,:changeMap.size(3)-kx]
        changeMapOut[0,0,:changeMap.size(2)-ky,:changeMap.size(3)-kx] += changeMap[0,0,ky:,kx:]
    changeMapOut = changeMapOut.gt(0).char()
    return changeMapOut

def changePropagation(changeMap, filtSize):
    assert(len(filtSize) == 2)
    if filtSize[0] == 1 and filtSize[1] == 1:
        return changeMap #no propagation for 1x1 filters...
    
    h, w = changeMap.size(-2), changeMap.size(-1)
    changeMapOut = changeMap.new(changeMap.size())
    block = (128,1,1)
    grid = ((h*w - 1)//block[0] + 1,1)
    
    C.changePropagation(np.int32(1), np.int32(grid[1]), np.int32(grid[0]), 
                      np.int32(block[2]), np.int32(block[1]), np.int32(block[0]), 
                      ffi.cast("bool *", changeMap.contiguous().data_ptr()),
                      ffi.cast("bool *", changeMapOut.contiguous().data_ptr()),
                      np.int32(w),
                      np.int32(h),
                      np.int32((filtSize[0]-1)/2),
                      np.int32((filtSize[1]-1)/2))
    return changeMapOut

def changePropagation_test1():
    
    #python impl. v. other composite implementation
    input, prevInput, threshold, filtSize = genTestData()
    expResp = changeDetection_python(input, prevInput, filtSize, threshold)
    changeMap = changeDetection_python(input, prevInput, (1,1), threshold)
    actResp = changePropagation_python(changeMap, filtSize)
    passed = (actResp == expResp).all()
    
    #python impl. v. gpu impl.
    input, prevInput, threshold, filtSize = genTestData()
    changeMap = changeDetection_python(input, prevInput, (1,1), threshold)
    expResp = changePropagation_python(changeMap.clone(), filtSize)
    actResp = changePropagation(changeMap.clone(), filtSize)
    passed = passed and (actResp == expResp).all()
    
    return passed




def changeIndexesExtr_python(changeMap):
    # method 1:
    nz = torch.nonzero(changeMap.view(-1)).int().view(-1)
    # method 2:
#    cm = changeMap.view(-1).byte()
#    global longCumSum
#    if ('longCumSum' not in globals()) or len(longCumSum) < len(cm):
#        longCumSum = torch.cuda.IntTensor().resize_(cm.size()).fill_(1).cumsum(0).add_(-1)
#    nz = longCumSum[:len(cm)].index(cm)
    return nz

def changeIndexesExtr(changeMap):
    nz = changeIndexesExtr_python(changeMap)
    return nz

def changeIndexesExtr_test1():
    cm = torch.CharTensor(25,32).zero_()
    cm = torch.CharTensor(129,254).zero_()

    ci_ref = []
    def insertChange(y,x):
        cm[y][x] = 1
        ci_ref.append(y*cm.size(-1)+x)
    insertChange(3,3)
    insertChange(7,5)
    insertChange(5,7)
    insertChange(7,1)
    insertChange(1,5)
    insertChange(24,31)

    ci_ref = torch.IntTensor(ci_ref)
    cm = cm.cuda()

#    ci_gpu, cnt_gpu = changeIndexesExtr(cm)
    ci = changeIndexesExtr_python(cm)
    passed = (ci_ref.sort()[0] == ci.cpu().sort()[0]).all()
    return passed


def genXMatrix(input, changeIndexes, filtSize, useHalf=False):
    inC, inH, inW = input.size(-3), input.size(-2), input.size(-1)
    kH, kW = filtSize
    numChanges = changeIndexes.numel()

    CUDA_NUM_THREADS = 128 #256
    threadZ = CUDA_NUM_THREADS//(kH*kW)
    block = (kW,threadZ,kH)
    grid = ((numChanges-1)//threadZ+1,1)

    XMatrix = input.new(numChanges, inC*kH*kW)

    if numChanges > 0: 
        CC = Chalf if useHalf else C
        CC.genXMatrix(np.int32(1), np.int32(grid[1]), np.int32(grid[0]), 
                         np.int32(block[2]), np.int32(block[1]), np.int32(block[0]), 
                         ffi.cast("float *", XMatrix.contiguous().data_ptr()),
                         ffi.cast("float *", input.contiguous().data_ptr()),
                         ffi.cast("int *", changeIndexes.contiguous().data_ptr()),
                         np.int32(kW), np.int32(kH),
                         np.int32(inC), np.int32(inW), np.int32(inH),
                         np.int32(numChanges))
    return XMatrix

def genXMatrix_python(input, changeIndexes, filtSize):
    assert(input.dim() == 4 and input.size(0) == 1)
    _, inC, inH, inW = input.size()
    kH, kW = filtSize
    numChanges = changeIndexes.numel()
    XMatrix = input.new(numChanges, inC*kH*kW)
    for changeIdxIdx, changeIdxPos in enumerate(changeIndexes[:numChanges]):
        iy = changeIdxPos // inW
        ix = changeIdxPos % inW
        for ky, kx in ((ky, kx) for ky in range(kH) for kx in range(kW)):
            y, x = iy + ky-(kH-1)//2, ix + kx-(kW-1)//2
            if 0 <= y and y < inH and 0 <= x and x < inW:
                for ic in range(inC):
                    v = input[0, ic, y, x]
                    XMatrix[changeIdxIdx, (ic*kH + ky)*kW + kx] = v
            else:
                for ic in range(inC):
                    XMatrix[changeIdxIdx, (ic*kH + ky)*kW + kx] = 0
    return XMatrix

def genXMatrix_test1():
    input, prevInput, threshold, filtSize = genTestData()
    changeMap = changeDetection_python(input, prevInput, filtSize, threshold)
    changeIndexes = changeIndexesExtr_python(changeMap)
    Xmatrix = genXMatrix(input.cuda(), changeIndexes.cuda(), filtSize)
    Xmatrix_cpu = genXMatrix_python(input, changeIndexes, filtSize)
    passed = (Xmatrix == Xmatrix_cpu).all()
    return passed

def updateOutput(YMatrix, changeIndexes, prevOutput, withReLU=False, useHalf=False):

    outC, outH, outW = prevOutput.size(-3), prevOutput.size(-2), prevOutput.size(-1)
    numChanges = changeIndexes.numel()
    CUDA_NUM_THREADS = 1024
    
    block=(CUDA_NUM_THREADS,1,1)
    grid=((numChanges*outC-1)//CUDA_NUM_THREADS + 1,1)

    if numChanges > 0: 
        CC = Chalf if useHalf else C
        CC.updateOutput(np.int32(1), np.int32(grid[1]), np.int32(grid[0]), 
                          np.int32(block[2]), np.int32(block[1]), np.int32(block[0]), 
                          ffi.cast("float *", YMatrix.contiguous().data_ptr()),
                          ffi.cast("float *", prevOutput.contiguous().data_ptr()),
                          ffi.cast("int *", changeIndexes.contiguous().data_ptr()),
                          np.int32(outW*outH),
                          np.int32(numChanges),
                          np.int32(outC),
                          np.byte(withReLU))
                            
    return prevOutput

def updateOutput_python(Ymatrix, changeIndexes, prevOutput, withReLU=False):
    numChanges = changeIndexes.numel()
    w = prevOutput.size(-1)
    assert(changeIndexes.size(0) == numChanges)
    for i, chngIdx in enumerate(changeIndexes):
        y = chngIdx // w
        x = chngIdx % w
        values = Ymatrix[:,i]
        if withReLU:
            values.clamp_(0, float('inf')) # 1e1000
        prevOutput[0,:,y,x] = values
    return prevOutput

def updateOutput_test1():
    nOut, h, w = 16, 5, 17
    numChanges = 15

    changeIndexes = torch.randperm(h*w)[:numChanges].int().cuda()
    Ymatrix = torch.randn(nOut, numChanges).cuda()
    prevOutput = torch.randn(nOut, h, w).cuda()

    output = updateOutput(Ymatrix.clone(), changeIndexes.clone(), prevOutput.clone(), withReLU=False)
    output_cpu = updateOutput_python(Ymatrix.clone(), changeIndexes.clone(), prevOutput.clone(), withReLU=False)

    passed = (output == output_cpu).all()
    return passed

def matrixMult_python(Xmatrix, weights, bias, activFun=None):
    if Xmatrix.numel() == 0:
        return Xmatrix.clone()
    
    Ymatrix = Xmatrix.matmul(weights.view(weights.size(0),-1).transpose(0,1))
    if bias is not None:
        Ymatrix.add_(bias) 
    if activFun is not None:
        Ymatrix = activFun(F.Variable(Ymatrix), inplace=True).data
    return Ymatrix


def overall_test1():

    #define params
    nOut, nIn, h, w = 16, 3, 20, 30
    kH, kW = filtSize = (3, 3)
    threshold = 2.25
    useRef = False

    #generate stimuli
    weights = torch.randn(nOut, nIn, kH, kW).cuda()
    bias = torch.randn(nOut).cuda()
    inputs = []
    inputs.append(torch.randn(nIn, h, w).cuda())
    inputs.append(inputs[0] + torch.randn(nIn, h, w).cuda())

    #init
    prevInput = torch.ones(nIn, h, w).cuda()*1e1000
    prevOutput = torch.randn(nOut, h, w).cuda()*1e1000

    for input in inputs:
        if useRef:
            changeMap = changeDetection_python(input, prevInput, filtSize, threshold)
            prevInput = input
            changeIndexes = changeIndexesExtr_python(changeMap)
            Xmatrix = genXMatrix_python(input, changeIndexes, filtSize)
            Ymatrix = matrixMult_python(Xmatrix, weights, bias)
            Ymatrix = Ymatrix.transpose(0,1)
            output = updateOutput_python(Ymatrix, changeIndexes, prevOutput)
        else:
            changeMap = changeDetection(input, prevInput, filtSize, threshold)
            prevInput = input
            changeIndexes = changeIndexesExtr_python(changeMap)
            Xmatrix = genXMatrix(input, changeIndexes, filtSize)
            Ymatrix = matrixMult_python(Xmatrix, weights, bias)
            Ymatrix = Ymatrix.transpose(0,1)
            output = updateOutput(Ymatrix, changeIndexes, prevOutput)

    return output

def runTests():
    print('change detection: %s' % ('passed' if changeDetection_test1()   else 'FAILED'))
    print('change idx extr.: %s' % ('passed' if changeIndexesExtr_test1() else 'FAILED'))
    print('genXmatrix:       %s' % ('passed' if genXMatrix_test1()        else 'FAILED'))
    print('update output     %s' % ('passed' if updateOutput_test1()      else 'FAILED'))
#runTests()

def getRuntime(func, numRepeat=10):

    import timeit
    import gc

    def runCore():
        func()
        torch.cuda.synchronize()

    #perform the measurement
    gc.disable()
    gc.collect()
    t = timeit.Timer(runCore)
    numRunsPerRepeat = 3
    for i in range(10):
        runCore()
    times = t.repeat(repeat=numRepeat, number=numRunsPerRepeat)
    bestTime = np.min(times)/numRunsPerRepeat
    gc.enable()
    return bestTime

def runBenchmarks():
    print('='*20)
    print('benchmarking results')
    print('-'*20)
    input, prevInput, threshold, filtSize = genTestData()
#    rt = getRuntime(lambda: changeDetection_python(input, prevInput, filtSize, threshold), numRepeat=1000)
    rt = getRuntime(lambda: changeDetection(input, prevInput, filtSize, threshold), numRepeat=1000)
    print('changeDetection [s]: %f' % (rt,))

    changeMap = changeDetection(input, prevInput, filtSize, threshold)
    rt = getRuntime(lambda: changeIndexesExtr_python(changeMap), numRepeat=100)
    print('changeIdxExtr.  [s]: %f' % (rt,))
    print('-'*20)
#runBenchmarks()

if __name__ == "__main__":   
#    passed = cbconvFG_test1(gpu=False)
#    if passed:
#        print('cpu: ok')
#    else:
#        print('cpu: FAIL')
#        assert(False) 
#    passed = cbconvFG_test1(gpu=True)
#    if passed:
#        print('gpu: ok')
#    else:
#        assert(False)
    
    ffi.dlclose(C)
