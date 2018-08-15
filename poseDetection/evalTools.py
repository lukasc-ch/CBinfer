#Copyright (c) 2018 ETH Zurich, Lukas Cavigelli
import torch
import tx2power
import math
import pycbinfer as cb

def inferFramesetBenchmark(m, frameset, cuda=True, numIter=3, preprocessor=None):
    #preprocess is necessary
    if preprocessor is not None:
        frameset = list(map(preprocessor, frameset))
        
    # copy data to GPU if necessary
    if cuda:
        frameset = [frm.cuda() for frm in frameset]
    
    #define benchmark/experiment setup
    def prepBenchm(): 
        cb.clearMemory(m)
        for frame in frameset[:-1]:
            m(torch.autograd.Variable(frame, volatile=True, requires_grad=False))
        if cuda:
            torch.cuda.synchronize()
    def coreBenchm(): 
        frame = frameset[-1]
        m(torch.autograd.Variable(frame, volatile=True, requires_grad=False))
        if cuda:
            torch.cuda.synchronize()
            
    import timeit
    tmr = timeit.Timer(stmt=coreBenchm, setup=prepBenchm)
    
    #run throughput measurements
    execTime = min(tmr.repeat(repeat=3, number=1))
    
    return execTime

def inferFrameset(m, frameset, cuda=True, preprocessor=None, postproc=None):
    if preprocessor is not None:
        frameset = list(map(preprocessor, frameset))
    if cuda:
        frameset = [frm.cuda() for frm in frameset]
    cb.clearMemory(m)
    
    for frame in frameset:
        y = m(torch.autograd.Variable(frame, volatile=True, requires_grad=False))
        
    if postproc is not None:
        y = postproc(y)
    return y
#    _, clsf = torch.max(y.cpu(), 1)
#    clsf = clsf[0].data
#    return clsf

def inferFramesetPowerMeasurement(m, frameset, cuda=True, numFrames=0, 
                                  preprocessor=None):
    if preprocessor is not None:
        frameset = list(map(preprocessor, frameset))
    if cuda:
        frameset = [frm.cuda() for frm in frameset]
        
    #create a longer sequence by going back-and-forth on the frames we have
    if numFrames > 0:
        frameset = frameset + frameset[-2:0:-1]
        frameset = math.ceil(numFrames/len(frameset))*frameset
        frameset = frameset[:numFrames]
    frame = frameset[0]
    cb.clearMemory(m)
    m(torch.autograd.Variable(frame, volatile=True, requires_grad=False))
    if cuda:
        torch.cuda.synchronize()
    pl = tx2power.PowerLogger(interval=1.0,#interval=0.05, 
                             nodes=tx2power.getNodesByName(nameList=[
                                     'board/main','module/main',
#                                     'module/cpu','module/gpu','module/ddr'
                                     ]))
    pl.start()
    for fid, frame in enumerate(frameset[1:]):
        m(torch.autograd.Variable(frame, volatile=True, requires_grad=False))
#        torch.cuda.synchronize()
#        pl.recordEvent('completed frame %d' % (fid,))
    torch.cuda.synchronize()
    pl.stop()
    return pl

def getCBconvLayers(model):
    cbModuleList = []
    print(list(sorted(model.named_children(), key=lambda m: m[0])))
    submodels = list(map(lambda m: m[1], sorted(model.named_children(), key=lambda m: m[0])))
    for submod in submodels:
        for m in submod:
            if type(m) is cb.conv2d.CBConv2d:
                cbModuleList.append(m)
    return cbModuleList

def getCBpoolLayers(model):
    cbModuleList = []
    print(list(sorted(model.named_children(), key=lambda m: m[0])))
    submodels = list(map(lambda m: m[1], sorted(model.named_children(), key=lambda m: m[0])))
    for submod in submodels:
        for m in submod:
            if type(m) is cb.conv2d.CBPoolMax2d:
                cbModuleList.append(m)
    return cbModuleList

def writeTable(table, appName='poseDet', seqName='sampleSequence', evalName='evalXX', 
               useHalf=None, disableRecursive=None, showPowerMode=False):
    import csv
    import os
#    filename = 'results/%s-%s-%s-%s-%s-%s.csv' % (
#            appName, os.uname().nodename, evalName, 
#            'half' if useHalf else 'single',
#            'recurFalse' if disableRecursive else 'recurTrue', seqName)
    filename = 'results/%s-%s-%s' % (appName, os.uname().nodename, evalName)
    if useHalf is not None:
        filename += '-' + ('half' if useHalf else 'single')
    if disableRecursive is not None:
        filename += '-' + ('recurFalse' if disableRecursive else 'recurTrue')
    if showPowerMode:
        import tx2power # check platform suppoertd
        if tx2power.powerSensorsPresent():
            filename += '-pwrMode' + tx2power.getPowerMode()
    filename += '-' + seqName + '.csv'
    
    with open(filename, 'w', newline='') as csvfile:
        csv.writer(csvfile).writerows(table)
