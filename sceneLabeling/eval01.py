#Copyright (c) 2018 ETH Zurich, Lukas Cavigelli
import torch
import torch.cuda
import pycbinfer
import types
import tx2power
import evalTools
from functools import reduce


#%% define postprocessor
def postproc(y):
    _, clsf = torch.max(y.cpu(), 1)
    clsf = clsf[0].data
    return clsf

#%% load models and convert to CBinfer
useCuda = True
useHalf = False
experimentIdx = 6

from modelLoader import loadModel
modelBaseline, modelTest = loadModel(experimentIdx, useCuda, useHalf)
print(modelTest)

#%% load data
import videoSequenceReader as vsr # select data loader and thus dataset
seqName = vsr.getSequenceNames(validationSetOnly=True)[1] # select frameset
frameset, target = vsr.getDataFrames(seqName=seqName, numFrames=5) # load frameset

#%% process and evaluate
clsfCB =  evalTools.inferFrameset(modelTest, frameset, cuda=useCuda, postproc=postproc)
clsfRef =  evalTools.inferFrameset(modelBaseline, frameset, cuda=useCuda, postproc=postproc)
accuracyRef = clsfRef.eq(target).sum() / target.numel()
accuracyCB  = clsfCB.eq(target).sum() / target.numel()
#accuracyRel = 1 - (clsfCB - clsfRef).ne(0).sum() / clsfRef.numel()
print('accuracy (reference): %2.2f%%' % (accuracyRef*100,))
print('accuracy (CBinfer):   %2.2f%%' % (accuracyCB *100,))
#print('accuracy (relative):  %2.2f%%' % (accuracyRel*100,))


#%% generate profiler output
#m = modelTest[3]
#x = torch.randn((1,3,240,320)).cuda()
#x = datasetFrames[0][0].cuda()
#y = model(x)
#_, clsf = torch.max(y.cpu(), 1)
#clsf = clsf[0].data
#scipy.misc.imshow(clsf.float().numpy())
#datasetTargets[0]
#torch.cuda.profiler.stop()


#%% throughput sweep evaluations
thresholdFactors = [f/5 for f in range(11)]
thresholdOneIdx = 5
numIter = 2 # 1 # 10 # number of measurement iterations
measurePower = True and tx2power.powerSensorsPresent()
numFramesForPowerMeasurement = 100

#prep work
cbconvModules = list(filter(lambda m: type(m) == pycbinfer.CBConv2d, modelTest.modules()))
cbconvThresholds = list(map(lambda m: m.threshold, cbconvModules))
seqNames = vsr.getSequenceNames(validationSetOnly=True)
print(seqNames)
#measure reference (cuDNN) throughput
frameset, target = vsr.getDataFrames(seqName=seqNames[0], numFrames=5)
execTimeRef = evalTools.inferFramesetBenchmark(modelBaseline, frameset[:1], numIter=numIter, cuda=useCuda)
#measure power
if measurePower:
    powerLogger = evalTools.inferFramesetPowerMeasurement(modelTest, frameset, 
                                                cuda=useCuda, numFrames=numFramesForPowerMeasurement)

#iterate over all sequences
resultsBySeq = []
for seqIdx, seqName in enumerate(seqNames):
    #get appropriate frame sequence
    frameset, target = vsr.getDataFrames(seqName=seqName, numFrames=5)
    #determine reference accuracy
    clsfRef =  evalTools.inferFrameset(modelBaseline, [frameset[-1]], cuda=useCuda, postproc=postproc)
    accuracyRef = clsfRef.eq(target).sum() / target.numel()
    
    resultsByThreshold = []
    for thIdx, thFactor in enumerate(thresholdFactors):
        print('processing seq. %d of %d and thFactor %d of %d' % 
              (seqIdx+1, len(seqNames), thIdx+1, len(thresholdFactors)))
        #configure model
        for th, m in zip(cbconvThresholds, cbconvModules): 
            m.threshold = thFactor*th
        result = types.SimpleNamespace(seqName=seqName, th=thFactor)

        #measure throughput
        result.execTimeRef = execTimeRef
        result.execTime = evalTools.inferFramesetBenchmark(modelTest, frameset, numIter=numIter, cuda=useCuda)
        
        #measure accuracy & stats
        for m in cbconvModules:
            m.gatherComputationStats = True
        result.accuracyRef = accuracyRef
        clsf = evalTools.inferFrameset(modelTest, frameset, cuda=useCuda, postproc=postproc)
        result.accuracy = clsf.eq(target).sum() / target.numel()
        result.compStats = list(map(lambda m: m.compStats, cbconvModules))
        for m in cbconvModules:
            m.gatherComputationStats = False
        
        #measure number of changes
        cbconvModules = list(filter(lambda m: type(m) == pycbinfer.CBConv2d, modelTest.modules()))
        for m in cbconvModules: 
            m.saveChangeMap = True
        evalTools.inferFrameset(modelTest, frameset, cuda=useCuda, postproc=postproc)
        changes = 0
        for m in cbconvModules: 
            if hasattr(m, 'changeMap'):
                changes += m.changeMap.sum()
                del m.changeMap
                m.saveChangeMap = False
            else:
                changes = float('-inf')
        result.changes = changes
        resultsByThreshold.append(result)
        
        #print current result
        if False:
            print('throughput (seq: %s, th:%0.2f) -- ref: %0.2fms (%0.2f fps), CB: %0.2fms (%0.2f fps), speed-up: %0.2fx' 
                  % (result.seqName, result.th, 
                     result.execTimeRef*1e3, 1/result.execTimeRef, 
                     result.execTime*1e3, 1/result.execTime, 
                     result.execTimeRef/result.execTime)) 
        
    resultsBySeq.append(resultsByThreshold)
    
#%% visualize
import matplotlib.pyplot as plt
plt.close('all')
for i, (seqName, resultsByThreshold) in enumerate(zip(seqNames, resultsBySeq)):
    #plot loss increase
    plt.figure(301, figsize=(14,4))
    plt.subplot(1,3,1)
    plt.title('loss increase (here: pixel-wise accuracy [%])')
    plt.grid(True)
    plt.plot(thresholdFactors, [(r.accuracyRef-r.accuracy)*100 for r in resultsByThreshold])
    #plot number of changes
    getNumOps = lambda r: reduce(lambda u,v: u+v, 
                                 map(lambda u: u['numInputPropedChanges'], 
                                     r.compStats))
    plt.subplot(1,3,2)
#    plt.title('number of changes')
    plt.title('number of operations')
    plt.grid(True)
    plt.semilogy(thresholdFactors, [getNumOps(r) for r in resultsByThreshold])
#    plt.semilogy(thresholdFactors, [r.changes for r in resultsByThreshold])
    #plot speed-up
    plt.subplot(1,3,3)
#    plt.title('speed-up')
#    plt.grid(True)
#    plt.plot(thresholdFactors, [r.execTimeRef/r.execTime for r in resultsByThreshold])
    plt.title('throughput (frame/s)')
    plt.grid(True)
    plt.plot(thresholdFactors, [1/r.execTime for r in resultsByThreshold])
    if i == len(seqNames)-1:
        cudnnThroughput = 1/resultsBySeq[0][0].execTimeRef
        plt.axhline(y=cudnnThroughput)
        ax1 = plt.gca()
        ax1.set_ylim(bottom=0)
        ax2 = ax1.twinx()
        ax2.set_label('speed-up')
        ax2.set_ylim(bottom=0, top=ax1.get_ylim()[1]/cudnnThroughput)

if measurePower:
    plt.figure(401, figsize=(14,4))
    powerLogger.showDataTraces()
    avgPwr = powerLogger.getAveragePower()
    energy = powerLogger.getTotalEnergy()
    print('NVPModel power mode: %s' % (tx2power.getPowerMode()))
    print('average power [mW]: %d' % (avgPwr))
    print('avg. energy [mJ/frame]: %d' % (energy/numFramesForPowerMeasurement))

#%%
if True:
    plt.figure(501, figsize=(14,4))
    seqIdx = 0
    numOps_CG = [[e['numInputPropedChanges'] for e in resByTh.compStats] for resByTh in resultsBySeq[seqIdx]]
    numOps_FG_FM = [[e['numInputPropedChangesPerFeatureMap'] for e in resByTh.compStats] for resByTh in resultsBySeq[seqIdx]]
    gainByFG_FM = [[e['numInputPropedChanges']/e['numInputPropedChangesPerFeatureMap'] for e in resByTh.compStats] for resByTh in resultsBySeq[seqIdx]]
    gainByFG_SP = [[e['numInputPropedChanges']/e['numInputChanges'] for e in resByTh.compStats] for resByTh in resultsBySeq[seqIdx]]
    gainByFG_FMSP = [[e['numInputPropedChanges']/e['numInputChangesPerFeatureMap'] for e in resByTh.compStats] for resByTh in resultsBySeq[seqIdx]]
    normalOPS = [[e['totalInputValues'] for e in resByTh.compStats] for resByTh in resultsBySeq[seqIdx]]
    totalGOPS = list(map(list, zip(*numOps_CG)))
    totalGOPS = totalGOPS
    
    plt.title('num. operations by layer and thFactor')
    plt.grid(True)
    plt.stackplot(thresholdFactors, list(map(list, zip(*numOps_CG)))) # transposed numOps_CG
#    plt.stackplot(thresholdFactors, list(map(list, zip(*numOps_FG_FM)))) # transposed numOps_FG_FM
    plt.legend(['Layer %d' % (i+1) for i in range(len(numOps_CG[0]))])
#    powerLogger.showDataTraces()


    plt.figure(503).clf()
    getOpTrace = lambda resSeq, typ: [reduce(lambda u,v: u+v, [e[typ] for e in resByTh.compStats]) for resByTh in resSeq]
    typeInt = ['totalInputValues', 'numInputPropedChanges', 'numInputChanges', 
               'numInputPropedChangesPerFeatureMap', 'numInputChangesPerFeatureMap']
    typeDescr = ['normal','CG','FG-spatial','FG-byFM','FG-byFM&SP']
    for t in typeInt:
        plt.plot(thresholdFactors, getOpTrace(resultsBySeq[seqIdx], t))
    plt.legend(typeDescr)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)

    plt.figure(504).clf()
    numSeq = len(seqNames)
    barWidth = (1-0.2)/numSeq
    tickPos = [p+0.5+0.1 for p in range(numSeq)]
    
    getOpBySeq = lambda resSeq, typ: [reduce(lambda u,v: u+v, [e[typ] for e in resSeq[thresholdOneIdx].compStats]) for resSeq in resultsBySeq]
    typeInt = ['totalInputValues', 'numInputPropedChanges', 'numInputChanges', 
               'numInputPropedChangesPerFeatureMap', 'numInputChangesPerFeatureMap']
    typeDescr = ['normal','CG','FG-spatial','FG-byFM','FG-byFM&SP']
    for i, t in enumerate(typeInt):
        plt.bar([p+(i+0.5+len(typeInt)//2)*barWidth for p in range(numSeq)], getOpBySeq(resultsBySeq, t), barWidth, bottom=0)
    plt.xticks(tickPos, seqNames)
    plt.legend(typeDescr)
    plt.ylim(ymin=0, ymax=2.5e10)
    plt.xlim(xmin=0)

#comparison at th=1 for all sequences
# - performance
#[1/s[thresholdOneIdx].execTime for s in resultsBySeq]
#1/resultsBySeq[0][thresholdOneIdx].execTimeRef
# - num ops
#[getNumOps(s[thresholdOneIdx])*1e-9 for s in resultsBySeq]
#refNumOps = reduce(lambda u,v: u+v, map(lambda u: u['totalInputValues'], 
#                                        resultsBySeq[0][0].compStats))*1e-9
    
#%% export files
exportFiles = False
if exportFiles:
#    from tabulate import tabulate
    import csv
    import os
    filename = 'results/sceneLabeling-%s-eval01-%s-%s.csv' % (
            os.uname().nodename, 'half' if useHalf else 'single', '%s')
    header = ['throughput', 'speedup']
    
    # write baseline values
    with open(filename % 'baseline', 'w', newline='') as csvfile:
        execTimeRef = resultsByThreshold[0].execTimeRef
        tbl = [[execTimeRef, 1.0]]
        csv.writer(csvfile).writerows([header]+tbl)
        
    #write file for different sequences
    header = ['th', 'loss', 'accPixel', 'numGOps', 'throughput', 'speedup', 'throughputBaseline']
    for i, (seqName, resultsByThreshold) in enumerate(zip(seqNames, resultsBySeq)):
        tbl = list(zip(*[thresholdFactors,
                         [(r.accuracyRef-r.accuracy)*100.0 for r in resultsByThreshold],
                         [r.accuracy*100.0 for r in resultsByThreshold],
#                         [(r.loss-r.lossRef) for r in resultsByThreshold],
                         [getNumOps(r)*1e-9 for r in resultsByThreshold],
                         [1/r.execTime for r in resultsByThreshold],
                         [execTimeRef/r.execTime for r in resultsByThreshold],
                         [1/execTimeRef for r in resultsByThreshold],
                         ]))
    #    tblStr = tabulate(tbl, header, 'grid') 
    #    print(tblStr)
        with open(filename % seqName, 'w', newline='') as csvfile:
            csv.writer(csvfile).writerows([header]+tbl)
    
    # 2nd plot with per-layer and by-threhold number of OPs
    seqIdx = 2
    seqName, resultsByTh = seqNames[seqIdx], resultsBySeq[seqIdx]
    numOps_CG = [[e['numInputPropedChanges'] for e in resByTh.compStats] for resByTh in resultsByTh]
    numOps_CG = list(zip(*numOps_CG))
    with open('results/sceneLabeling-eval02-%s.csv' % seqName, 'w', newline='') as csvfile:
        csv.writer(csvfile).writerows(zip(*([thresholdFactors] + numOps_CG)))

