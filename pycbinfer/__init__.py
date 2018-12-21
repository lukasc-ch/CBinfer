#Copyright (c) 2018 ETH Zurich, Lukas Cavigelli

import torch
import torch.cuda
import torch.nn as nn

from .conv2d import CBConv2d
from .conv2d import CBPoolMax2d

def subsitute(node, threshold=1e-1, finegrained=False):
    print(type(node))
    if type(node) is torch.nn.modules.conv.Conv2d:
        if node.dilation == (1,1) and node.stride == (1,1):
            print('replacing conv2d')
            m = CBConv2d(node, threshold)
            m.finegrained = finegrained
            return m, True
    return node, False


def convertRecur(m, ignoreList=[], threshold=1e-1, finegrained=False): # ignoreList= [model.model_cpu_net.Lambda]
    changed = False

    mout = nn.Sequential()
    for i, (nodeName, node) in enumerate(m.named_children()):
        
        if type(node) in [nn.Sequential]:
            # handle nn.Sequential containers through recursion
            m, c = convertRecur(node, ignoreList, threshold)
            mout.add_module(nodeName, m)
            changed |= c
        elif type(node) in ignoreList + [nn.Dropout]:
            # remove nodes not needed during inference (e.g. Dropout, and those in the ignoreList)
            print('removing node %s' % (type(node),))
            changed = True
            continue
        else:
            # handle simple substitutions (i.e. convert Conv2d to CBconv2d)
            nodeOut, newNode = subsitute(node, threshold=threshold, finegrained=finegrained)
            mout.add_module(nodeName, nodeOut)
            changed |= newNode

    # another round until convergence?
    if changed:
        mout = convert(mout, ignoreList)
    return mout, changed

def mergeReLURecur(m):
    mout = nn.Sequential()
    for i, (nodeName, node) in enumerate(m.named_children()):
        
        # handle nn.Sequential containers through recursion
        if type(node) in [nn.Sequential]:
            mout.add_module(nodeName, mergeReLURecur(node))
            continue
        # enable built-in ReLU of CBconv
        elif type(node) in [CBConv2d]:
            chldrn = list(m.children())
            if len(chldrn) > i+1 and type(chldrn[i+1]) is torch.nn.modules.activation.ReLU:
                node.withReLU = True
        # remove ReLU if CBconv layer proceeded
        elif type(node) is torch.nn.modules.activation.ReLU and i >= 1 and type(list(m.children())[i-1]) is CBConv2d:
            print('merging ReLU layer')
            continue # i.e. don't add the module!!
        
        mout.add_module(nodeName, node)
    return mout

def propChangeIndexesOf1x1(rootModule):
    seqContainers = list(filter(lambda m: type(m) == torch.nn.Sequential, rootModule.modules()))
    for seqCont in seqContainers:
        mPrev = None
        for m in seqCont:
            if type(m) == CBConv2d and type(mPrev) == CBConv2d and m.kernel_size == [1,1]:
                print('enabling propagation of change indexes for 1x1')
                mPrev.propChangeIndexes = True
            mPrev = m
    return rootModule

def clearMemory(net):
    for m in net.modules():
        if type(m) == CBConv2d or type(m) == CBPoolMax2d:
            m.clearMemory()

def getStateTensors(net):
    state = []
    for m in net.modules():
        if type(m) == CBConv2d or type(m) == CBPoolMax2d:
            state += m.getStateTensors()
    return state

def convert(m, ignoreList=[], threshold=1e-1): # ignoreList= [model.model_cpu_net.Lambda]
    m1, changed = convertRecur(m, ignoreList=ignoreList, threshold=threshold)
    mout = mergeReLURecur(m1)
    return mout



def tuneThresholdParameters(vidSeqReader, evalSequences, numFramesPerSeq, 
                            targetGenerator, preprocessor, 
                            modelBaseline, modelTest, evaluator, 
                            cbModuleList, lossToleranceList, initThreshold=1e-2, thresholdIncrFactor=1.2):
        
    #obtain MUT (model under test) and MREF (model reference)
    #for all sequences in dataset:
    # load sequence
    # preprocess
    # apply MUT to sequence, apply MREF to last frame
    # compare/evaluate
    
    if type(lossToleranceList) is not list:
        # if loss tolerance is given as a single value, apply it to all modules
        lossToleranceList = [lossToleranceList]*len(cbModuleList)
    assert(len(cbModuleList)==len(lossToleranceList))
    
    def evaluateModel():
        clearMemory(modelTest)
        totalLoss = 0
        for seqName in evalSequences:
            frames, target = vidSeqReader.getDataFrames(seqName=seqName, numFrames=numFramesPerSeq)
            if target is None:
                target = targetGenerator(frames[-1])
            for frame in frames:
                feedData = preprocessor(frame)
                outTest = modelTest(torch.autograd.Variable(feedData, volatile=True).cuda())
            loss = evaluator(outTest, target)
            totalLoss += loss
        return totalLoss
    
    #greedy front-to-back threshold adjustment
    prevLoss = evaluateModel()
    for i, m in enumerate(cbModuleList):
        print('adjusting threshold for module %d of %d' % (i+1, len(cbModuleList)))
        m.threshold = initThreshold # initialize threshold value
        while True:
            # increase th while loss ok. Once insufficient, take 1 step back.
            m.threshold *= thresholdIncrFactor 
            loss = evaluateModel()
            print('. (%f < %f + %f)' % (loss,prevLoss,lossToleranceList[i],))
            if loss - prevLoss > lossToleranceList[i]:
                m.threshold /= thresholdIncrFactor 
                break
        prevLoss = evaluateModel()
        
    print('to save the model, run: pycbinfer.clearMemory(modelTest); torch.save(modelTest, "./models/modelXXX.net")')
    
    #problem: model gets worse and worse; should error mostly be introduced at the beginning?! Should the back-off be larger?
    # gets stuck at 13/14th modules, i.e. when getting into the non-VGGnet layers.... are they ultra-robust???!!
    # can we make the assmption that prev. threshold is good start of next threshold???
    #evaluate profit per layer as threshold decision criterion!!! I.e., delta-MSE/delta-changeCount

