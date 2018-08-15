#Copyright (c) 2018 ETH Zurich, Lukas Cavigelli
import math
import torch
import tx2power
import pycbinfer as cb
from functools import reduce
import matplotlib.pyplot as plt
import evalTools
from poseDetEvaluator import PoseDetectionEvaluator


#%% load models and apply network configuration
experimentIdx = 11
useHalf = True #False
useCuda = True
disableRecursive = False#True#False#True
thFactor = 1.0#1.5#1.0
numFrames = 200
seqName = 'caviar-walkByShop1cor-seq000-5xSubsample'
#seqName = '2DiQUX11YaY-720p-seq000-twoDancers'
#seqName = '2DiQUX11YaY-720p-seq004-twoDancers-hfr'


#%% load model
modelTest, modelBaseline = PoseDetectionEvaluator.getModels(experimentIdx=experimentIdx)
poseDetEvaluator = PoseDetectionEvaluator(modelTest, modelBaseline, useCuda=useCuda, useHalf=useHalf)

# obtain sequential list of CBconv modules
cbconvModules = evalTools.getCBconvLayers(modelTest)

#enable/disable recursive mode
if disableRecursive:
    print('WARNING: non-recursive version')
    for m in cbconvModules:
        m.feedbackLoop = False
        
#apply threshold factor
cbconvThresholds = list(map(lambda m: m.threshold, cbconvModules))
for th, m in zip(cbconvThresholds, cbconvModules): 
    m.threshold = thFactor*th

#%% load data
import videoSequenceReader as vsr # select data loader and thus dataset
frameset, target = vsr.getDataFrames(seqName=seqName, numFrames=numFrames)
#apply preprocessing
frameset = list(map(poseDetEvaluator.preprocessor, frameset))

#%% trace 1 & 2: accuracy / loss & no. operations
cb.clearMemory(modelTest)
for m in cbconvModules:
    m.gatherComputationStats = True
    
lossTrace, opsTrace, outpVarTrace = [], [], []
for t, frm in enumerate(frameset):
    outpRef = poseDetEvaluator.modelApply(frm, modelBaseline)
    outpAct = poseDetEvaluator.modelApply(frm, modelTest)
    loss = poseDetEvaluator.evaluator([outpAct], [outpRef], method='maxAbsDiff') # non_MSE metric is being used (!!)
#    loss = poseDetEvaluator.evaluator([outpAct], [outpRef], method='mse') 
    lossTrace.append(loss)
    outpVarTrace.append(outpRef.var())
    numOps = reduce(lambda u,v: u+v, map(lambda m: m.compStats['numInputPropedChanges'], cbconvModules))
    opsTrace.append(numOps)
outpStdDev = math.sqrt(reduce(lambda u,v: u+v, outpVarTrace)/len(frameset))
#lossTraceRel = list(map(lambda v: v/outpStdDev, lossTrace))
numOpsBL = reduce(lambda u,v: u+v, map(lambda m: m.compStats['totalInputValues'], cbconvModules))
opsTraceRel = list(map(lambda v: v/numOpsBL, opsTrace))

for m in cbconvModules:
    m.gatherComputationStats = False

plt.figure(3)
plt.clf()
#plt.plot(lossTraceRel)
plt.plot(lossTrace)
plt.grid()
plt.ylabel('loss')
plt.xlabel('time (frame index)')

plt.figure(5)
plt.clf()
plt.plot(opsTraceRel)
plt.grid()
plt.ylabel('ref. num. operations')
plt.xlabel('time (frame index)')

#%% trace 3: exec. time
def inferNextFrameBenchmark(model, frame, numIter=3):
    stateCopy = [t.clone() for t in cb.getStateTensors(model)]
    frame = frame.cuda()
    torch.cuda.synchronize()
    
    #define benchmark/experiment setup
    def prepBenchm(): 
        for tNow, tPrev in zip(cb.getStateTensors(model), stateCopy):
            tNow.copy_(tPrev)
        torch.cuda.synchronize()
    def coreBenchm(): 
        model(torch.autograd.Variable(frame, volatile=True, requires_grad=False))
        torch.cuda.synchronize()
    import timeit
    
    tmr = timeit.Timer(stmt=coreBenchm, setup=prepBenchm)
    
    #run throughput measurements
    execTime = min(tmr.repeat(repeat=3, number=1))
    return execTime
    

for t in cb.getStateTensors(modelTest):
    t.fill_(-1e100)
timeTrace = []
for t, frm in enumerate(frameset):
    timeTrace.append(inferNextFrameBenchmark(modelTest, frm, numIter=3))
timeBaseline = inferNextFrameBenchmark(modelBaseline, frameset[-1], numIter=3)
timeSpeedup = list(map(lambda t: timeBaseline/t, timeTrace))
plt.figure(7)
plt.clf()
plt.plot(timeTrace)
plt.grid()
plt.ylabel('runtime [s] per frame')
plt.xlabel('time (frame index)')
color = 'tab:red'
plt.twinx()
plt.plot(timeSpeedup, color=color)
plt.ylabel('speed-up', color=color)

#%% trace 4: power
#measurePower = True and tx2power.powerSensorsPresent()
powerTrace = []
frameIndexes = list(range(0,numFrames-1,10)) 
for frmIdx in frameIndexes:
    frm1 = frameset[frmIdx].cuda()
    frm2 = frameset[frmIdx].cuda()
    for i in range(10): #needs to be large, lots of change!!
        modelTest(torch.autograd.Variable(frm1, volatile=True, requires_grad=False))
        modelTest(torch.autograd.Variable(frm2, volatile=True, requires_grad=False))
    torch.cuda.synchronize()
    pwr = tx2power.getModulePower()
    powerTrace.append(pwr)
    
for i in range(10): #needs to be large, lots of change!!
    modelBaseline(torch.autograd.Variable(frm1, volatile=True, requires_grad=False))
    modelBaseline(torch.autograd.Variable(frm2, volatile=True, requires_grad=False))
torch.cuda.synchronize()
pwrBaseline = tx2power.getModulePower()
    
plt.figure(9).clf()
plt.plot(frameIndexes, powerTrace)
plt.grid()
plt.ylabel('module power [mW]')
plt.xlabel('time (frame index)')
plt.axhline(y=pwrBaseline)

energyTrace = [timeTrace[tIdx]*powerTrace[pIdx] for pIdx, tIdx in enumerate(frameIndexes)]
energyBaseline = pwrBaseline*timeBaseline
plt.figure(10).clf()
plt.plot(frameIndexes, energyTrace)
plt.grid()
plt.ylabel('energy per frame [mJ]')
plt.xlabel('time (frame index)')
plt.axhline(y=energyBaseline)


#%% write file
numFrames = len(timeTrace)
evalTools.writeTable([['frameIdx', 'loss', 'numGOps', 'numOpsRel', 
                       'numGOpsBL', 'execTime', 'speedup']] +
                     list(zip(range(numFrames), lossTrace, [v*1e-9 for v in opsTrace], opsTraceRel, 
                              [numOpsBL*1e-9]*numFrames, timeTrace, timeSpeedup)), 
                     appName='poseDet', seqName=seqName, evalName='eval03', 
                     useHalf=useHalf, disableRecursive=disableRecursive,
                     showPowerMode=True)

numEnergySamples = len(frameIndexes)
evalTools.writeTable([['frameIdx', 'power', 'execTime', 'energy', 'powerBL', 'execTimeBL', 'energyBL']] +
                     list(zip(frameIndexes, 
                              powerTrace, 
                              [timeTrace[tIdx] for tIdx in frameIndexes], 
                              energyTrace, 
                              [pwrBaseline]*numEnergySamples, 
                              [timeBaseline]*numEnergySamples, 
                              [energyBaseline]*numEnergySamples)), 
                     appName='poseDet', seqName=seqName, evalName='eval04', 
                     useHalf=useHalf, disableRecursive=disableRecursive,
                     showPowerMode=True)
                     
                     
