#Copyright (c) 2018 ETH Zurich, Lukas Cavigelli
import torch
import pycbinfer
torch.set_num_threads(torch.get_num_threads())

print('warming up')

# load baseline model
modelBaseline = torch.load('models/modelBaseline.net').eval().cuda()

#generate/convert CBconv model
experimentIdx = 6
modelTest = pycbinfer.convert(modelBaseline)#, threshold=1e-2)
if experimentIdx >= 0 and experimentIdx < 4:
    pass
elif experimentIdx >= 4:
    # EXPERIMENT 4: Enable feedback loop (+ Exp. 3)
    #    modelTest = torch.nn.Sequential(*[modelTest[i] for i in range(5)] + [modelBaseline[i] for i in range(8,11)])
    #    for m in list(filter(lambda m: type(m) == pycbinfer.CBConv2d, modelTest.modules())):
    #        m.copyInput = False
    cbconvModules = list(filter(lambda m: type(m) == pycbinfer.CBConv2d, 
                                modelTest.modules()))
    for m in cbconvModules:
        m.feedbackLoop = True

#define sequential list of CBconv modules
cbModuleList = []
print(list(sorted(modelTest.named_children(), key=lambda m: int(m[0]))))
submodels = list(map(lambda m: m[1],
                     sorted(modelTest.named_children(),
                            key=lambda m: int(m[0]))))
for m in submodels:
    if type(m) is pycbinfer.conv2d.CBConv2d:
        cbModuleList.append(m)
        
# define evaluation function (i.e. loss function / value to minimize)
def evaluator(outTest, target):
    outTest = outTest.cpu()
    _, clsfTest = outTest.data[0].max(0)
    errorRate = target.ne(clsfTest).sum() / target.numel()
    return errorRate

#define data set and data loader; define or generate ground truth
import videoSequenceReader as vidSeqReader

def targetGenerator(frame):
    #not needed for this dataset; ground truth is known and not generated
    raise NotImplementedError()
    
def preprocessor(im):
    return im
    
    
if __name__ == "__main__":
    print('converting model to CBinfer and tuning threshold parameters')
    pycbinfer.tuneThresholdParameters(
            vidSeqReader, 
            evalSequences=vidSeqReader.getSequenceNames(validationSetOnly=True), 
            numFramesPerSeq=5, 
            targetGenerator=targetGenerator, preprocessor=preprocessor, 
            modelBaseline=modelBaseline, modelTest=modelTest, evaluator=evaluator, 
            cbModuleList=cbModuleList, 
            lossToleranceList=[0.0005] + [0.00025]*(len(cbModuleList)-1),
            initThreshold=1e-2, 
            thresholdIncrFactor=1.2)
