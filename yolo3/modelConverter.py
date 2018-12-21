#Copyright (c) 2018 ETH Zurich, Lukas Cavigelli
# This script performs the mapping from a DNN to CBinfer and setting the thresholds

import torch
import pycbinfer
import detect 
import util
from darknet import Darknet
#import openPose as op
torch.set_num_threads(torch.get_num_threads())


print('warming up')

#%% DEFINE PARAMETERS & BASELINE MODEL
#   ==================================
#T = 2
#modelBaseline = op.PoseModel.fromFile(T=T).cuda()
args = detect.arg_parse()
scales = args.scales
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)

num_classes = 80
classes = util.load_classes('data/coco.names') 



#%% DEFINE MODEL UNDER TEST
# =======================

#load network
def loadModel():
    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")
    
    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32
    #scales = [int(x) for x in scales.split(',')]
    #scale_inds = modelTest.get_scale_inds(scales, inp_dim)
    model = model.cuda()
    model.eval()
    return model


modelBaseline = loadModel()
modelTest = loadModel()

#modify network
moduleList = modelTest.module_list
for i in range(len(moduleList)):
    m = moduleList[i]
    assert(type(m) == torch.nn.Sequential)
    moduleList[i] = pycbinfer.convert(m)
    #TODO: merge conv2d and BN(!!!)
modelTest.timed = False


print(modelTest)


#modelTest = op.PoseModel.fromFile(T=T)
#modelTest.timed = False#True # enable priting timing results
#modelTest.model0   = pycbinfer.convert(modelTest.model0, threshold=1e-2)
#modelTest.model1_1 = pycbinfer.convert(modelTest.model1_1)
#modelTest.model1_2 = pycbinfer.convert(modelTest.model1_2)
#modelTest.model2_1 = pycbinfer.convert(modelTest.model2_1)
#modelTest.model2_2 = pycbinfer.convert(modelTest.model2_2)
#if T != 2: print('not all submodels converted to CBinfer!!')
#modelTest = modelTest.cuda()
#poseDetTest = op.PoseDetector(modelTest)


#%% OLD

# obtain sequential list of CBconv modules
def getCBModuleListTwolevel(modelTest):
    cbModuleList = []
    print(list(sorted(modelTest.named_children(), key=lambda m: m[0])))
    submodels = list(map(lambda m: m[1],sorted(modelTest.named_children(), key=lambda m: m[0])))
    try:
        for submod in submodels:
                for m in submod:
                    if type(m) is pycbinfer.CBConv2d: #pycbinfer.conv2d.CBConv2d
                        cbModuleList.append(m)
        return cbModuleList
    except TypeError as e:
        pass
def getCBModuleList(model):
    return [m for m in model.modules() if type(m) is pycbinfer.CBConv2d] #pycbinfer.conv2d.CBConv2d
    
#cbModuleList = getCBModuleListTwolevel(modelTest)
cbModuleList = getCBModuleList(modelTest)


#%% DEFINE SETUP (PREPROC, MODEL APPL., EVALUATION, DATA SET)
#   =========================================================
#@lru_cache(maxsize=100)
def preprocessor(img):
    return img.unsqueeze(0)#torch.from_numpy(img).float().cuda()#img#poseDetTest.preprocess(img).unsqueeze(0)

def modelApply(inp, model):
    global paf, heatmap
    inVar = torch.autograd.Variable(inp).cuda()
    outVar = model(inVar)
    return outVar.data#torch.cat([paf.data, heatmap.data], dim=-3)

def evaluator(outTest, target):
    outTest = outTest.data.cpu()#torch.cat(outTest, dim=-3).data.cpu() # pafTest, heatmapTest = outTest
    target = target.data.cpu()#torch.cat(target, dim=-3)
    diff = outTest - target
    mse = diff.pow_(2).mean()
    return mse

import videoSequenceReader as vidSeqReader

def targetGenerator(frame):
    feedData = preprocessor(frame)
    target = modelApply(feedData, modelBaseline).cpu()
    return target
    





#%% MODIFY NETWORK FOR EXPERIMENTS

# modify model to test / experiments
for m in cbModuleList:
    m.feedbackLoop = True
       
    
#%% RUN THRESHOLD SELECTION
#   =======================
experimentIdx = 1
    #%% - LINEARIZED
if experimentIdx <= 10:
    pycbinfer.tuneThresholdParameters(
            vidSeqReader, 
    #        evalSequences=['2DiQUX11YaY-720p-seq000-twoDancers-hfr'], 
            evalSequences=['seq01'], 
            numFramesPerSeq=10, 
            targetGenerator=targetGenerator, preprocessor=preprocessor, 
            modelBaseline=modelBaseline, modelTest=modelTest, evaluator=evaluator, 
            cbModuleList=cbModuleList, 
            lossToleranceList=[5e-5]*(1 if len(cbModuleList)>0 else 0) + [2e-6]*(len(cbModuleList)-1),
            initThreshold=5e-2, 
            thresholdIncrFactor=1.2)

    #%% - HIERARCHICAL
elif experimentIdx == 11:
        
    def tuneModules(modelTest, cbModuleList, lossToleranceList):
            pycbinfer.tuneThresholdParameters(
                vidSeqReader, 
        #        evalSequences=['2DiQUX11YaY-720p-seq000-twoDancers-hfr'], 
                evalSequences=['2DiQUX11YaY-720p-seq000-twoDancers'], 
                numFramesPerSeq=10, 
                targetGenerator=targetGenerator, preprocessor=preprocessor, 
                modelBaseline=modelBaseline, modelTest=modelTest, evaluator=evaluator, 
                cbModuleList=cbModuleList, 
                lossToleranceList=lossToleranceList,
                initThreshold=5e-2, 
                thresholdIncrFactor=1.2)
        
    def getThresholds(moduleList):
        return [m.threshold for m in moduleList]
        
    def setThresholds(moduleList, thresholdList):
        # if threholdList is a single value, apply it for all elements
        if type(thresholdList) is not list:
            thresholdList = [thresholdList]*len(moduleList)
        assert(len(moduleList) == len(thresholdList))
        
        for m, th in zip(moduleList, thresholdList):
            m.threshold = th
    
    for m in cbModuleList:
        m.threshold = 0
        
        
    lossTolFirst = 5e-5
    lossTolDefault = 2e-6
        
    # tune model0 and model1_1    
    cbModuleListTmp = getCBModuleList(modelTest.model0) + getCBModuleList(modelTest.model1_1)
    tuneModules(modelTest, cbModuleListTmp, lossTolFirst + lossTolDefault*(len(cbModuleListTmp)-1))
    
    # store an clear threshold for model1_1 for later restoring
    cbModuleList_1_1 = getCBModuleList(modelTest.model1_1)
    thresholds_1_1 = getThresholds(cbModuleList_1_1)
    setThresholds(cbModuleList_1_1, 0)
    # tune model1_2
    cbModuleListTmp = getCBModuleList(modelTest.model1_2)
    tuneModules(modelTest, cbModuleListTmp, lossTolDefault)
    #restore thresholds of model1_1
    setThresholds(cbModuleList_1_1, thresholds_1_1)
    
    # tune model2_1    
    cbModuleListTmp = getCBModuleList(modelTest.model2_1)
    tuneModules(modelTest, cbModuleListTmp, lossTolDefault)
    
    # store an clear threshold for model2_1 for later restoring
    cbModuleList_2_1 = getCBModuleList(modelTest.model2_1)
    thresholds_2_1 = getThresholds(cbModuleList_2_1)
    setThresholds(cbModuleList_2_1, 0)
    # tune model2_2
    cbModuleListTmp = getCBModuleList(modelTest.model2_2)
    tuneModules(modelTest, cbModuleListTmp, lossTolDefault)
    #restore thresholds of model2_1
    setThresholds(cbModuleList_2_1, thresholds_2_1)

else:
    assert(False)


#torch.save(modelTest, 'cbModelTest.pt')
#torch.save(modelBaseline, 'cbModelBaseline.pt')

#visualize first change map: plt.imshow(poseModel.model0[0].changeMap.cpu().float().numpy())

#visualize = False # needs to be False for nvvp profiling...
#visualize = True
#torch.cuda.synchronize()
#for frameIdx, frame in enumerate(frames):
#    print('processing frame %i' % (frameIdx,))
#    
#    canvas = poseDet.detectAndInpaint(frame)
#
#    # Display the resulting frame
#    if visualize:
#        plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
#        plt.draw()
##        if 'fig2' not in locals() or fig2 is None:
##            fig2 = plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
##        else:
##            fig2.set_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
#        plt.pause(0.01)
