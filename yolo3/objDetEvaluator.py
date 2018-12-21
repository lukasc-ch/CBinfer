#Copyright (c) 2018 ETH Zurich, Lukas Cavigelli
import torch
import torch.cuda
import pycbinfer
#import openPose as op

class ObjDetEvaluator:

    def __init__(self, modelTest, modelBaseline, useCuda, useHalf):
        
        if useCuda:
            modelBaseline = modelBaseline.cuda()
            modelTest = modelTest.eval().cuda()
        else:
            modelBaseline = modelBaseline.cpu()
            modelTest = modelTest.eval().cpu()
        if useHalf:
            modelTest = modelTest.half()
            modelBaseline = modelBaseline.half()
            
#        self.poseDetTest = op.PoseDetector(modelTest)
        self.modelBaseline = modelBaseline
        self.useCuda = useCuda
        self.useHalf = useHalf
    
    @staticmethod
    def getModels(experimentIdx):
#        assert(experimentIdx==None)
        modelBaseline = torch.load('cbModelBaseline.pt')#op.PoseModel.fromFile(T=2).eval()
        modelTest = torch.load('cbModelTest.pt')
#        if experimentIdx <= 8:
#            modelTest = torch.load('models/model3.net')
#        elif experimentIdx == 9:
#            modelTest = torch.load('models/model009.net')
#        elif experimentIdx == 10:
#            modelTest = torch.load('models/model010.net') # experiment 10 -- trained for recursive
#        elif experimentIdx == 11:
#            modelTest = torch.load('models/model011c.net') # recursive AND with optimized threshold selection
#            # a: straight-forward, b: ???, c: no pre-init of thresholds for whole block
#        else:
#            assert(False)
        pycbinfer.clearMemory(modelTest)
        
        return modelTest, modelBaseline
    
    
    def preprocessor(self, img):
#        tmp = self.poseDetTest.preprocess(img).unsqueeze(0)
        tmp = img.unsqueeze(0)
        if self.useHalf: 
            tmp = tmp.half()
        return tmp
    
    @staticmethod
    def modelApply(inp, model):
        inVar = torch.autograd.Variable(inp).cuda()
        outVar = model(inVar)
        return outVar.data
    
    @staticmethod
    def evaluator(outTest, target, method='mse'):
        #MSE
        outTest = outTest.float().cpu()
        target = target.float().cpu()
        
#        outTest = [t.float() for t in outTest]
#        outTest = torch.cat(outTest, dim=-3).cpu()
#        target = [t.float() for t in target]
#        target = torch.cat(target, dim=-3).cpu()
        
        if not(torch.is_tensor(outTest)):
            outTest = outTest.data # pafTest, heatmapTest = outTest
            
        if method == 'mse':
            loss = (outTest-target).pow_(2).mean()
        elif method == 'maxAbsDiff':
            loss = (outTest-target).abs_().max()
        else:
            assert(False)
            
        return loss
    
#    @staticmethod
#    def evaluator3(outTest, target):
#        #max(abs((outp-ref)/abs(ref)))
#        outTest = [t.float() for t in outTest]
#        outTest = torch.cat(outTest, dim=-3).cpu()
#        target = [t.float() for t in target]
#        target = torch.cat(target, dim=-3).cpu()
#        
#        if not(torch.is_tensor(outTest)):
#            outTest = outTest.data # pafTest, heatmapTest = outTest
#        diff = (outTest - target)/(target.abs()+1e-5)
#        maxAbsDiff = diff.abs_().max()
#        return maxAbsDiff
    
    def targetGenerator(self, frame):
        feedData = self.preprocessor(frame)
        target = self.modelApply(feedData, self.modelBaseline).cpu()
        return target
