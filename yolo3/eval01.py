#Copyright (c) 2018 ETH Zurich, Lukas Cavigelli
import types
import tx2power
from functools import reduce
import evalTools
import objDetEvaluator
#evaluator = objDetEvaluator.ObjDetEvaluator
##from poseDetEvaluator import PoseDetectionEvaluator

import torch

#%% load models and convert to CBinfer
experimentIdx = 1
useHalf = False
useCuda = True

modelTest, modelBaseline = objDetEvaluator.ObjDetEvaluator.getModels(experimentIdx=experimentIdx)
evaluator = objDetEvaluator.ObjDetEvaluator(modelTest, modelBaseline, useCuda=useCuda, useHalf=useHalf)

# obtain sequential list of CBconv modules
def getCBconvLayers(model):
    return list(filter(lambda m: str(type(m)) == "<class 'pycbinfer.conv2d.CBConv2d'>", 
                       model.modules()))
#cbconvModules = evalTools.getCBconvLayers(modelTest)
cbconvModules = getCBconvLayers(modelTest)
print(modelTest)


#%% load data
import videoSequenceReader as vsr # select data loader and thus dataset
seqName = vsr.getSequenceNames(validationSetOnly=True)[1] # select frameset
frameset, target = vsr.getDataFrames(seqName=seqName, numFrames=10) # load frameset


#%% throughput sweep evaluations
#sweep parameters
thresholdFactors = [f/5 for f in range(21)]
thresholdOneIdx = 5

numIter = 1#2 # number of measurement iterations
measurePower = True and tx2power.powerSensorsPresent()
numFramesForPowerMeasurement = 100
cbconvThresholds = list(map(lambda m: m.threshold, cbconvModules))
#seqNames = vsr.getSequenceNames(validationSetOnly=True)
seqNames = ['seq01']#, 'seq02']
#measure reference (cuDNN) throughput
frameset, target = vsr.getDataFrames(seqName=seqNames[0], numFrames=5)
execTimeRef = evalTools.inferFramesetBenchmark(modelBaseline, frameset, numIter=numIter, 
                                               cuda=useCuda, preprocessor=evaluator.preprocessor)
#measure power
if measurePower:
    powerLogger = evalTools.inferFramesetPowerMeasurement(modelTest, frameset, 
                                                          cuda=useCuda, numFrames=numFramesForPowerMeasurement, 
                                                          preprocessor=evaluator.preprocessor)

#iterate over all sequences
resultsBySeq = []
for seqIdx, seqName in enumerate(seqNames):
    #get appropriate frame sequence
    frameset, target = vsr.getDataFrames(seqName=seqName, numFrames=5)
    #determine reference accuracy
    clsfRef =  evalTools.inferFrameset(modelBaseline, [frameset[-1]], cuda=useCuda, 
                                       preprocessor=evaluator.preprocessor)
    if target is None:
        target = evaluator.targetGenerator(frameset[-1])
        lossRef = 0
    else:
        #there is no real reference if there is not target...
        lossRef = evaluator.evaluator(clsfRef, target)
    
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
        result.execTime = evalTools.inferFramesetBenchmark(modelTest, frameset, numIter=numIter, 
                                                           cuda=useCuda, preprocessor=evaluator.preprocessor)
        
        #measure accuracy & stats
        for m in cbconvModules:
            m.gatherComputationStats = True
#        result.accuracyRef = accuracyRef
        result.lossRef = lossRef
        clsf = evalTools.inferFrameset(modelTest, frameset, cuda=useCuda, 
                                       preprocessor=evaluator.preprocessor)
#        result.accuracy = clsf.eq(target).sum() / target.numel()
        result.loss = evaluator.evaluator(clsf, target)
        result.compStats = list(map(lambda m: m.compStats, cbconvModules))
        for m in cbconvModules:
            m.gatherComputationStats = False
        
        #measure number of changes
#        cbconvModules = list(filter(lambda m: type(m) == pycbinfer.CBConv2d, modelTest.modules()))
#        for m in cbconvModules: 
#            m.saveChangeMap = True
#        evalTools.inferFrameset(modelTest, frameset, cuda=useCuda, preprocessor=preprocessor)
#        changes = 0
#        for m in cbconvModules: 
#            if hasattr(m, 'changeMap'):
#                changes += m.changeMap.sum()
#                del m.changeMap
#                m.saveChangeMap = False
#            else:
#                changes = float('-inf')
#        result.changes = changes
        
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
cudnnLineStyle = {'color':'black', 'linestyle':'dashed'}
for i, (seqName, resultsByThreshold) in enumerate(zip(seqNames, resultsBySeq)):
    #plot loss increase
    plt.figure(301, figsize=(14,4))
    plt.subplot(1,4,1)
    plt.title('loss increase')
    plt.grid(True)
    plt.plot(thresholdFactors, [(r.loss-r.lossRef) for r in resultsByThreshold])
    
    #plot number of operations
    plt.subplot(1,4,2)
    plt.title('number of operations')
    plt.grid(True)
    getNumOps = lambda r: reduce(lambda u,v: u+v, 
                                 map(lambda u: u['numInputPropedChanges'], 
                                     r.compStats), 0)
    numOpsTable = [getNumOps(r) for r in resultsByThreshold]
    plt.semilogy(thresholdFactors, numOpsTable)
    if i == len(seqNames)-1:
        getFullEvalNumOps = reduce(lambda u,v: u+v, 
                                   map(lambda u: u['totalInputValues'], 
                                       resultsByThreshold[0].compStats), 0)
        plt.axhline(y=getFullEvalNumOps, **cudnnLineStyle)
        ax1 = plt.gca()
        ax1.set_ylim(bottom=0)
        ax2 = ax1.twinx()
#        ax2.set_ylim(bottom=0, top=ax1.get_ylim()[1]/getFullEvalNumOps)
        ax2.set_ylim(bottom=0, top=getFullEvalNumOps)
    
    #plot speed-up
    plt.subplot(1,4,3)
    plt.title('thrghpt [frm/s] and speed-up v th.')
    plt.grid(True)
    plt.plot(thresholdFactors, [1/r.execTime for r in resultsByThreshold])
    if i == len(seqNames)-1:
        cudnnThroughput = 1/execTimeRef
        plt.axhline(y=cudnnThroughput, **cudnnLineStyle)
        ax1 = plt.gca()
        ax1.set_ylim(bottom=0)
        ax2 = ax1.twinx()
        ax2.set_ylim(bottom=0, top=ax1.get_ylim()[1]/cudnnThroughput)
        
    #plot loss v. threshold
    plt.subplot(1,4,4)
    plt.title('thrghpt [frm/s] / speed-up for varying th')
    plt.grid(True)
    plt.plot([r.loss-r.lossRef for r in resultsByThreshold], [1/r.execTime for r in resultsByThreshold])
    if i == len(seqNames)-1:
        cudnnThroughput = 1/execTimeRef
        plt.axhline(y=cudnnThroughput, **cudnnLineStyle)
        ax1 = plt.gca()
        ax1.set_ylim(bottom=0)
        ax2 = ax1.twinx()
        ax2.set_ylim(bottom=0, top=ax1.get_ylim()[1]/cudnnThroughput)
    plt.legend(seqNames)
        
#%% power and energy
print('Sequences: ')
for i, seqName in enumerate(seqNames):
    print(' %d: %s' % (i+1, seqName))

if measurePower:
#    plt.figure(401, figsize=(14,4))
#    powerLogger.showDataTraces()
    avgPwr = powerLogger.getAveragePower()
    energy = powerLogger.getTotalEnergy()
    print('NVPModel power mode: %s' % (tx2power.getPowerMode()))
    print('average power [mW]: %d' % (avgPwr))
    print('avg. energy [mJ/frame]: %d' % (energy/numFramesForPowerMeasurement))

#%% num. operations per layers
if True:
    plt.figure(501, figsize=(14,4))
    seqIdx = -1
    numOps_CG = [[e['numInputPropedChanges'] for e in resByTh.compStats] for resByTh in resultsBySeq[seqIdx]]
    numOps_FG_FM = [[e['numInputPropedChangesPerFeatureMap'] for e in resByTh.compStats] for resByTh in resultsBySeq[seqIdx]]
#    gainByFG_FM = [[e['numInputPropedChanges']/e['numInputPropedChangesPerFeatureMap'] for e in resByTh.compStats] for resByTh in resultsBySeq[seqIdx]]
#    gainByFG_SP = [[e['numInputPropedChanges']/e['numInputChanges'] for e in resByTh.compStats] for resByTh in resultsBySeq[seqIdx]]
#    gainByFG_FMSP = [[e['numInputPropedChanges']/e['numInputChangesPerFeatureMap'] for e in resByTh.compStats] for resByTh in resultsBySeq[seqIdx]]
    normalOPS = [[e['totalInputValues'] for e in resByTh.compStats] for resByTh in resultsBySeq[seqIdx]]
    totalGOPS = list(map(list, zip(*numOps_CG)))
    totalGOPS = totalGOPS
    
    plt.title('num. operations by layer and thFactor')
    plt.grid(True)
    plt.stackplot(thresholdFactors, list(map(list, zip(*numOps_CG)))) # transposed numOps_CG
#    plt.stackplot(thresholdFactors, list(map(list, zip(*normalOPS)))) # transposed numOps_CG
#    plt.stackplot(thresholdFactors, list(map(list, zip(*numOps_FG_FM)))) # transposed numOps_FG_FM
    plt.legend(['Layer %d' % (i+1) for i in range(len(numOps_CG[0]))])
#    powerLogger.showDataTraces()


    # Operations by threshold for CG v. various FG
    plt.figure(503).clf()
    getOpTrace = lambda typ: [reduce(lambda u,v: u+v, [e[typ] for e in resByTh.compStats]) for resByTh in resultsBySeq[seqIdx]]
    typeInt = ['totalInputValues', 'numInputPropedChanges', 'numInputChanges', 
               'numInputPropedChangesPerFeatureMap', 'numInputChangesPerFeatureMap']
    typeDescr = ['normal','CG','FG-spatial','FG-byFM','FG-byFM&SP']
    for t in typeInt:
        plt.plot(thresholdFactors, getOpTrace(t))
    plt.legend(typeDescr)
    plt.ylim(ymin=0)
    plt.xlim(xmin=0)


#    # Operations by sequence (th=1) for CG v. various FG
#    plt.figure(504).clf()
#    numSeq = len(seqNames)
#    tickPos = [p+0.5+0.1 for p in range(numSeq)]
#    getOpBySeq = lambda typ: [reduce(lambda u,v: u+v, [e[typ] for e in resSeq[thresholdOneIdx].compStats]) for resSeq in resultsBySeq]
#    typeInt = ['totalInputValues', 'numInputPropedChanges', 'numInputChanges', 
#               'numInputPropedChangesPerFeatureMap', 'numInputChangesPerFeatureMap']
#    typeDescr = ['normal','CG','FG-spatial','FG-byFM','FG-byFM&SP']
#    barWidth = (1-0.2)/len(typeDescr)
#    for i, t in enumerate(typeInt):
#        plt.bar([p+(i+0.5+len(typeInt)//2)*barWidth for p in range(numSeq)], getOpBySeq(t), barWidth, bottom=0)
#    plt.xticks(tickPos, seqNames)
#    plt.legend(typeDescr)
#    plt.ylim(ymin=0)#, ymax=1e11)
#    plt.xlim(xmin=0)


    # Operations savings by layer (fixed seq., th=1) for CG v. various FG
    plt.figure(505).clf()
    getOpByLayer = lambda typ: [e[typ]/1e9 for e in resultsBySeq[seqIdx][thresholdOneIdx].compStats]
    typeInt = ['totalInputValues', 'numInputPropedChanges', 'numInputChanges', 
               'numInputPropedChangesPerFeatureMap', 'numInputChangesPerFeatureMap']
    typeDescr = ['normal','CG','FG-spatial','FG-byFM','FG-byFM&SP']
    for i, t in enumerate(typeInt):
        numops = getOpByLayer(t)
        plt.plot(range(len(numops)), numops)
    plt.legend(typeDescr)
    plt.xlabel('layer idx')
    plt.ylabel('# GOp/frame')
    plt.ylim(ymin=0)#, ymax=1e11)
    
    
    
#%% export files
exportFiles = False
if exportFiles:
    from tabulate import tabulate
    import csv
    import os
    filename = 'results/objDet-%s-eval01-%s-%s.csv' % (
            os.uname().nodename, 'half' if useHalf else 'single', '%s')
    header = ['throughput', 'speedup']
    # write baseline values
    with open(filename % 'baseline', 'w', newline='') as csvfile:
        execTimeRef = resultsByThreshold[0].execTimeRef
        tbl = [[execTimeRef, 1.0]]
        csv.writer(csvfile).writerows([header]+tbl)
        
    #write file for different sequences
    header = ['th', 'lossIncr', 'numGOps', 'throughput', 'speedup', 'throughputBaseline']
    for i, (seqName, resultsByThreshold) in enumerate(zip(seqNames, resultsBySeq)):
        tbl = list(zip(*[thresholdFactors,
                         [(r.loss-r.lossRef) for r in resultsByThreshold],
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
    with open('results/eval02-%s.csv' % seqName, 'w', newline='') as csvfile:
        csv.writer(csvfile).writerows(zip(*([thresholdFactors] + numOps_CG)))

    # Operations savings by layer (fixed seq., th=1) for CG v. various FG
    getOpByLayer = lambda typ: [e[typ]/1e9 for e in resultsBySeq[seqIdx][thresholdOneIdx].compStats]
    typeInt = ['totalInputValues', 'numInputPropedChanges', 'numInputChanges', 
               'numInputPropedChangesPerFeatureMap', 'numInputChangesPerFeatureMap']
    typeDescr = ['normal','CG','FG-SP','FG-FM','FG-FMandSP']
    tbl = [getOpByLayer(t) for t in typeInt]
    tbl = list(zip(*([range(len(tbl[0]))] + tbl)))
    header = ['layerIdx'] + typeDescr
    evalTools.writeTable([header]+tbl, appName='objDet', seqName=seqName, evalName='eval02b', 
                         useHalf=None, disableRecursive=None, showPowerMode=False)

