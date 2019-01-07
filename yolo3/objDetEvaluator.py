#Copyright (c) 2018 ETH Zurich, Lukas Cavigelli
import torch
import torch.cuda
import pycbinfer
import util
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
    def evaluator(outTest, target, method='objDet-coco-objectness'):
#    def evaluator(outTest, target, method='mse'):
        #MSE
        outTest = outTest.float().cpu()
        target = target.float().cpu()
        
#        outTest = [t.float() for t in outTest]
#        outTest = torch.cat(outTest, dim=-3).cpu()
#        target = [t.float() for t in target]
#        target = torch.cat(target, dim=-3).cpu()
        
#        if not(torch.is_tensor(outTest)):
#            outTest = outTest.data # pafTest, heatmapTest = outTest
            
        if method == 'mse':
            loss = (outTest-target).pow_(2).mean()
        elif method == 'maxAbsDiff':
            loss = (outTest-target).abs_().max()
        elif method == 'objDet-coco-objectness':
            loss = (outTest[...,0]-target[...,0]).pow_(2)[target[...,0] >= 0.5].mean()
#            loss = (outTest[...,0]-target[...,0]).abs_()[target[...,0] >= 0.5].mean()
#            loss = (outTest[...,0]-target[...,0]).abs_().mean()
        elif method == 'objDet-coco-iou':
            confidence = 0.5
            nms_thresh = 0.4
            num_classes = 80
#            prediction = prediction[:,scale_inds]
            predOut = util.write_results(outTest.data.cpu(), confidence, num_classes, nms=True, nms_conf=nms_thresh)
            predTarget = util.write_results(target.data.cpu(), confidence, num_classes, nms=True, nms_conf=nms_thresh)
            
            iouSum = 0
            interSum, unionSum = 0, 0
            for t in predTarget:
                tx0, ty0, tx1, ty1 = t[1:5]
                targetIoU, targetIoUIdx = 0, -1
                bestInterArea, bestUnionArea = 0, 0
                for i, o in enumerate(predOut):
                    x0, y0, x1, y1 = o[1:5]
                    
                    inter_x0 =  torch.max(tx0, x0)
                    inter_y0 =  torch.max(ty0, y0)
                    inter_x1 =  torch.min(tx1, x1)
                    inter_y1 =  torch.min(ty1, y1)
    
                    #intersection area
                    inter_area = (torch.max(inter_x1 - inter_x0 + 1, torch.zeros_like(inter_x1))*
                                  torch.max(inter_y1 - inter_y0 + 1, torch.zeros_like(inter_y1)))
                    
                    #union area diff
                    b1_area = (tx1 - tx0 + 1)*(ty1 - ty0 + 1)
                    b2_area = ( x1 -  x0 + 1)*( y1 -  y0 + 1)
                    union_area = b1_area + b2_area - inter_area
                    
                    iou = inter_area / union_area
                    if iou >= targetIoU:
                        targetIoU = iou
                        targetIoUIdx = i
                        bestInterArea = inter_area
                        bestUnionArea = union_area
                iouSum += targetIoU
                interSum += bestInterArea
                unionSum += bestUnionArea
                
            return 1/(interSum/unionSum)
#            return 1/iouSum
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
