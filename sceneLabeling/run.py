#Copyright (c) 2018 ETH Zurich, Lukas Cavigelli
import torch
import torch.cuda
import evalTools

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
modelRef, modelCB = loadModel(experimentIdx, useCuda, useHalf)
print(modelCB)

#%% load data
framesetIdx = 1

import videoSequenceReader as vsr
seqName = vsr.getSequenceNames(validationSetOnly=True)[framesetIdx]
frameset, target = vsr.getDataFrames(seqName=seqName, numFrames=5)

#%% process and evaluate
clsfCB =  evalTools.inferFrameset(modelCB, frameset, cuda=useCuda, postproc=postproc)
clsfRef =  evalTools.inferFrameset(modelRef, [frameset[-1]], cuda=useCuda, postproc=postproc)
accuracyRef = clsfRef.eq(target).sum() / target.numel()
accuracyCB  = clsfCB.eq(target).sum() / target.numel()
#accuracyRel = 1 - (clsfCB - clsfRef).ne(0).sum() / clsfRef.numel()
print('accuracy (reference): %2.2f%%' % (accuracyRef*100,))
print('accuracy (CBinfer):   %2.2f%%' % (accuracyCB *100,))
#print('accuracy (relative):  %2.2f%%' % (accuracyRel*100,))


torch.cuda.profiler.stop()
