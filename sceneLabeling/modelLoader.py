#Copyright (c) 2018 ETH Zurich, Lukas Cavigelli
import torch
import pycbinfer

def loadModel(experimentIdx, useCuda=True, useHalf=False):
    """Loads the baseline model and the model to test/perform experiments on. 
    
    experimentIdx:
        1. skipping change detection for last 2 layers (1x1 conv)
        2. no CBinfer for 1x1 layers
        3. not copying input anymore when not necessary (i.e. proceeding layer is not CBinfer) AND no CBinfer for 1x1 layers
        4. Enable feedback loop (+ Exp. 3)
        5. change-based pooling (+ Exp. 4)
        6. re-optimized/-converted model, otherwise as in Exp. 5
        7. fine-grained convolution (+ Exp. 3)
    """
    
    assert useHalf == False
    
    modelBaseline = torch.load('models/modelBaseline.net').eval()
    if useCuda:
        modelBaseline = modelBaseline.cuda()
    else:
        modelBaseline = modelBaseline.cpu()
    
    
    if experimentIdx <= 3:
        modelTest = torch.load('models/model003.net').eval()
    elif experimentIdx <= 5:
        #model with feedback loop
        modelTest = torch.load('models/model005.net').eval()
    else:
        modelTest = torch.load('models/model006.net').eval()
    if useCuda:
        modelTest = modelTest.cuda()
    else:
        modelTest = modelTest.cpu()
    #modelTest = pycbinfer.propChangeIndexesOf1x1(modelTest)
    
    #model modifications
    if experimentIdx == 1:
        # EXPERIMENT 1: change indexes propagation
        modelTest[4].propChangeIndexes=True
        modelTest[5].propChangeIndexes=True
    elif experimentIdx == 2:
        # EXPERIMENT 2: no CBinfer for 1x1 layers
        modelTest = torch.nn.Sequential(*[modelTest[i] for i in range(5)] + [modelBaseline[i] for i in range(8,11)])
    elif experimentIdx == 3:
        # EXPERIMENT 3: not copying input anymore when not necessary (i.e. proceeding layer is not CBinfer) AND no CBinfer for 1x1 layers
        modelTest = torch.nn.Sequential(*[modelTest[i] for i in range(5)] + [modelBaseline[i] for i in range(8,11)])
        for m in list(filter(lambda m: type(m) == pycbinfer.CBConv2d, modelTest.modules())):
            m.copyInput = False
        # can be done like this here, but generally only if preceeding module is not inplace.
    elif experimentIdx == 4:
        # EXPERIMENT 4: Enable feedback loop (+ Exp. 3)
        modelTest = torch.nn.Sequential(*[modelTest[i] for i in range(5)] + [modelBaseline[i] for i in range(8,11)])
        for m in list(filter(lambda m: type(m) == pycbinfer.CBConv2d, modelTest.modules())):
            m.copyInput = False
        cbconvModules = list(filter(lambda m: type(m) == pycbinfer.CBConv2d, modelTest.modules()))
        for m in cbconvModules:
            m.feedbackLoop = True
    elif experimentIdx == 5 or experimentIdx == 6:
        # EXPERIMENT 5: change-based pooling (+ Exp. 4)
        # EXPERIMENT 6: re-optimized model, otherwise as in Exp. 5
        modelTest = torch.nn.Sequential(*[modelTest[i] for i in range(5)] + [modelBaseline[i] for i in range(8,11)])
        for m in list(filter(lambda m: type(m) == pycbinfer.CBConv2d, modelTest.modules())):
            m.copyInput = False
        cbconvModules = list(filter(lambda m: type(m) == pycbinfer.CBConv2d, modelTest.modules()))
        for m in cbconvModules:
            m.feedbackLoop = True
        # replace pooling with change-based pooling layers and enable change indexes propagation
        modelTest[0].propChangeIndexes = True
        modelTest_1 = pycbinfer.CBPoolMax2d(modelTest[1]).cuda()
        modelTest[2].copyInput = True # no effect, since we are in feedbackLoop mode anyway...
        modelTest[2].propChangeIndexes = True
        modelTest_3 = pycbinfer.CBPoolMax2d(modelTest[3]).cuda()
        modelTest[4].copyInput = True
        modelTest = torch.nn.Sequential(*[modelTest[0], modelTest_1, modelTest[2], modelTest_3] + list(modelTest.children())[4:])
    elif experimentIdx == 7:
        # EXPERIMENT 77: fine-grained convolution (incl. stuff from before)
        modelTest = torch.nn.Sequential(*[modelTest[i] for i in range(5)] + [modelBaseline[i] for i in range(8,11)])
        for m in list(filter(lambda m: type(m) == pycbinfer.CBConv2d, modelTest.modules())):
            m.copyInput = False
            #enable fine-grained convolution of all CBconv layers
        cbconvModules = list(filter(lambda m: type(m) == pycbinfer.CBConv2d, modelTest.modules()))
        for m in cbconvModules:
            m.finegrained = True
    
    return modelBaseline, modelTest
