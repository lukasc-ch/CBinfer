#Copyright (c) 2018 ETH Zurich, Lukas Cavigelli
import torch 
import cv2
import pycbinfer as cb
import openPose as op
import videoSequenceReader as vsr
import matplotlib.pyplot as plt
torch.set_num_threads(8)


#default values
gpu = False
visualize = False
loadModelPath = None

gpu = True
visualize = True # turn off for profiling
useCBinfer = False # automatically true, if CBinfer-based model is loaded
#loadModelPath = './models/model2.net' 
#loadModelPath = './models/model3.net' 
#loadModelPath = './models/model009.net' 
loadModelPath = './models/model010.net' 
loadModelPath = './models/modelBaseline.net' 
#loadModelPath = './models/model3_recur.net' # implies using CBinfer or not based on file with the trained thresholds
thresholdFactor = 1.5#3.5 not-ok #2.5 ok mostly #1.5 ok #1.0 ok # 2.0 ok
#measureTiming = False
measureTiming = True
numFrame = 10 # 12 # 20
#useCBinfer = True 
seqName = 'ski-seq000'
seqName = '2DiQUX11YaY-720p-seq000-twoDancers'
#seqName = 'r62ttrftU4w-720p-seq002-perron2'

#def quickPlot(img):
#    import numpy as np
#    import matplotlib.pyplot as plt
#    import torch
#    if type(img) == np.ndarray:
#        assert(len(img.shape) == 3)
#        assert(img.shape[-1] in [1,3])
#        implt = img
#    elif type(img) in [torch.FloatTensor, torch.ByteTensor, torch.CharTensor, 
#                       torch.cuda.FloatTensor, torch.cuda.ByteTensor, torch.cuda.CharTensor]:
#        implt = img.cpu().float()
##        if implt.max() > 1.0:
##            implt /= 256.0
#        implt -= (implt.min() - 0.001)
#        implt /= (implt.max() + 0.001)
#        if implt.dim() == 4:
#            implt = implt[0]
#        implt = implt.transpose(0,1).transpose(1,2)
#        implt = implt.contiguous().numpy()
#    else:
#        assert(False)
#    plt.imshow(implt)
    

print('warming up')
#_ = handle_one(np.ones((320,320,3)))
#fig = None
#fileOrCamId = 0 #'sample_image/youtube_2DiQUX11YaY_360p.avi'# 0 for webcam
#if 'video_capture' not in globals() or not(video_capture.isOpened()):
#    video_capture = cv2.VideoCapture(fileOrCamId)
    
# instantiate pose model and pose detector
if loadModelPath is not None:
    poseModel = torch.load(loadModelPath)
    for m in poseModel.modules():
        if type(m) == cb.CBConv2d:
            m.threshold *= thresholdFactor
else:
    poseModel = op.PoseModel.fromFile(T=2)
#    poseModel.timed = True # enable priting timing results
#    finegrained = True
    if useCBinfer:
        poseModel.model0   = cb.convert(poseModel.model0)
        poseModel.model1_1 = cb.convert(poseModel.model1_1)
        poseModel.model1_2 = cb.convert(poseModel.model1_2)
        poseModel.model2_1 = cb.convert(poseModel.model2_1)
        poseModel.model2_2 = cb.convert(poseModel.model2_2)
poseModel.timed = measureTiming
if gpu: 
    poseModel.cuda()
else:
    poseModel.cpu()
cb.clearMemory(poseModel)
poseDet = op.PoseDetector(poseModel)

#poseModel.model0[0].finegrained = True
#poseModel.model0[0].threshold *= 0.1

#visualize first change map: plt.imshow(poseModel.model0[0].changeMap.cpu().float().numpy())
# print run time: poseModel.timed = True

frames, _ = vsr.getDataFrames(seqName=seqName, numFrames=numFrame)
if gpu: 
    torch.cuda.synchronize()


##ANALYSIS 1: visualize several changeMaps
#changeMapsVis = [str(v) for v in [0,2,12,25]]
#for cmi in changeMapsVis:
#    dict(poseModel.model0.named_children())[cmi].saveChangeMap = True


##ANALYSIS 2: fine-grained analysis computation statistics
#gatherComputationStats
#for m in poseModel.modules():
#    if type(m) == cb.CBConv2d:
#        m.gatherComputationStats = True
        
##ANALYSIS 3: speed-up by propagating 1x1 changes
#cb.propChangeIndexesOf1x1(poseModel)

#ANALYSIS 4: fine-grained conv
#modulesConvFG = (
#                list(poseModel.model0.modules()) + 
#                list(poseModel.model1_1.modules()) + 
#                list(poseModel.model1_2.modules()) +
#                list(poseModel.model2_1.modules()) + 
#                list(poseModel.model2_2.modules()) +
#                [])
#for m in modulesConvFG: 
#    if type(m) == cb.CBConv2d:
#        m.finegrained = True
        
#poseModel.model0[0].finegrained = True
#poseModel.model0[3].finegrained = True
#poseModel.model0[4].finegrained = True
#poseModel.model0[5].finegrained = True
        
        
#ANALYSIS 5: NOT compying the input all the time. 
# can skip it, if the previous model already copies the input (i.e. non-inplace modules)
# only occurs for level 1
selMods = list(poseModel.model0.children())
for prv, nxt in list(zip(selMods[:-1], selMods[1:])):
    if type(nxt) == cb.CBConv2d and type(prv) != cb.CBConv2d:
        nxt.copyInput = False


def processAndDisplay(frame):
#    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    canvas = poseDet.detectAndInpaint(frame, gpu=gpu)

    # Display the resulting frame
    if visualize:
        plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        plt.draw()
        plt.pause(0.01)
    
#import torch.autograd
#with torch.autograd.profiler.profile() as prof:
frameIdx = 0
print('processing frame %i' % (frameIdx,))
processAndDisplay(frames[0])
for frame in frames[1:]:
    frameIdx += 1
    print('processing frame %i' % (frameIdx,))
    processAndDisplay(frame)


##ANALYSIS 0: overall runtime measurement
#%timeit poseDet.detectAndInpaint(frames[0]); poseDet.detectAndInpaint(frames[5])


##ANALYSIS 1: visualize several changeMaps
#for i, cmi in enumerate(changeMapsVis):
#    cm = dict(poseModel.model0.named_children())[cmi].changeMap.cpu()
#    rate = cm.ne(0).sum() / cm.numel()
#    plt.figure(100 + i)
#    plt.title('layer %s (%2.2d%%)' % (cmi,rate*100))
#    plt.imshow(cm.float().numpy())

##ANALYSIS 2: fine-grained evaluation
#def printStats(model, name, sep='.'):
#    splt = name.split(sep)
#    m = model
#    for s in splt: 
#        m = dict(m.named_children())[s]
#    print('----------------')
#    print('layer: %s' % (name,))
#    for k,v in m.compStats.items():
#        print('%s: %s' % (str(k), str(v)))
#    global cs
#    cs = copy.deepcopy(m.compStats)
#    tiv = cs['totalInputValues']
#    del cs['totalInputValues']
#    for k,v in cs.items():
#        print('%s: %d3.1x' % (str(k), float(tiv)/float(v)))
#printStats(poseModel, 'model0.0')
#printStats(poseModel, 'model0.12')
#printStats(poseModel, 'model1_1.8')
#printStats(poseModel, 'model1_2.8')
#printStats(poseModel, 'model2_1.8')
#printStats(poseModel, 'model2_2.8')
##conclusion: gain very small at input, but quite large deeper in the network

#pycuda.driver.stop_profiler()
torch.cuda.profiler.stop()
