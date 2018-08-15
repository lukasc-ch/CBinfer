#Copyright (c) 2018 ETH Zurich, Lukas Cavigelli
import torch
import torch as T
import torch.nn as nn
import torchvision
from .util import *
import os

POSEMODEL_PATH_DEFAULT = os.path.join(os.path.dirname(__file__), './model/pose_model.pth')

class PoseModel(nn.Module):

    @classmethod
    def fromFile(cls, weightFilePath=POSEMODEL_PATH_DEFAULT, T=2):


        def make_layers(cfg_dict):
            layers = []
            for i in range(len(cfg_dict)-1):
                one_ = cfg_dict[i]
                for k,v in one_.items():
                    if 'pool' in k:
                        layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2] )]
                    else:
                        conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
                        layers += [conv2d, nn.ReLU(inplace=True)]
            one_ = list(cfg_dict[-1].keys())
            k = one_[0]
            v = cfg_dict[-1][k]
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
            layers += [conv2d]
            return nn.Sequential(*layers)

        block0 = [{'conv1_1':[3,64,3,1,1]}, {'conv1_2':[64,64,3,1,1]}, {'pool1_stage1':[2,2,0]},
                  {'conv2_1':[64,128,3,1,1]},{'conv2_2':[128,128,3,1,1]},{'pool2_stage1':[2,2,0]},
                  {'conv3_1':[128,256,3,1,1]},{'conv3_2':[256,256,3,1,1]},{'conv3_3':[256,256,3,1,1]},
                  {'conv3_4':[256,256,3,1,1]},{'pool3_stage1':[2,2,0]},
                  {'conv4_1':[256,512,3,1,1]},{'conv4_2':[512,512,3,1,1]},
                  {'conv4_3_CPM':[512,256,3,1,1]},{'conv4_4_CPM':[256,128,3,1,1]}]
        layers = []
        for i in range(len(block0)):
            one_ = block0[i]
            for k,v in one_.items():
                if 'pool' in k:
                    layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2] )]
                else:
                    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
                    layers += [conv2d, nn.ReLU(inplace=True)]


        blocks = {}
        blocks['block1_1']  = [{'conv5_1_CPM_L1':[128,128,3,1,1]},
              {'conv5_2_CPM_L1':[128,128,3,1,1]},{'conv5_3_CPM_L1':[128,128,3,1,1]},
              {'conv5_4_CPM_L1':[128,512,1,1,0]},{'conv5_5_CPM_L1':[512,38,1,1,0]}]

        blocks['block1_2']  = [{'conv5_1_CPM_L2':[128,128,3,1,1]},
              {'conv5_2_CPM_L2':[128,128,3,1,1]},{'conv5_3_CPM_L2':[128,128,3,1,1]},
              {'conv5_4_CPM_L2':[128,512,1,1,0]},{'conv5_5_CPM_L2':[512,19,1,1,0]}]

        for i in range(2,7):
            blocks['block%d_1'%i]  = [{'Mconv1_stage%d_L1'%i:[185,128,7,1,3]},
                  {'Mconv2_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv3_stage%d_L1'%i:[128,128,7,1,3]},
                  {'Mconv4_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv5_stage%d_L1'%i:[128,128,7,1,3]},
                  {'Mconv6_stage%d_L1'%i:[128,128,1,1,0]},{'Mconv7_stage%d_L1'%i:[128,38,1,1,0]}]
            blocks['block%d_2'%i]  = [{'Mconv1_stage%d_L2'%i:[185,128,7,1,3]},
                  {'Mconv2_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv3_stage%d_L2'%i:[128,128,7,1,3]},
                  {'Mconv4_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv5_stage%d_L2'%i:[128,128,7,1,3]},
                  {'Mconv6_stage%d_L2'%i:[128,128,1,1,0]},{'Mconv7_stage%d_L2'%i:[128,19,1,1,0]}]

        models = {}
        models['block0']=nn.Sequential(*layers)
        for k,v in blocks.items():
            models[k] = make_layers(v)


        # create and load model
        model = PoseModel(models, T=T)
        model.load_state_dict(torch.load(weightFilePath))
        
        # remove unnecessary layers
        #[m for name, m in poseModel.named_children() if name.startswith('model%d' % (3,))]
#        print('TODO: remove unnecessary layers')
        for name in ['model%d_%d' % (i,ext) for i in range(T+1,6+1) for ext in [1,2]]:
            delattr(model, name)
        
        # set the right mode
#        model.cuda()
        model.float()
        model.eval()
        return model

    def __init__(self,model_dict, transform_input=False, T=6):
        super(PoseModel, self).__init__()
        assert(T >= 1 and T <= 6)
        self.T = T

        self.model0   = model_dict['block0']
        self.model1_1 = model_dict['block1_1']
        self.model2_1 = model_dict['block2_1']
        self.model3_1 = model_dict['block3_1']
        self.model4_1 = model_dict['block4_1']
        self.model5_1 = model_dict['block5_1']
        self.model6_1 = model_dict['block6_1']

        self.model1_2 = model_dict['block1_2']
        self.model2_2 = model_dict['block2_2']
        self.model3_2 = model_dict['block3_2']
        self.model4_2 = model_dict['block4_2']
        self.model5_2 = model_dict['block5_2']
        self.model6_2 = model_dict['block6_2']
        
        self.timed = False

    def getModels(self):
        modelF = self.model0
#        modelsL = [self.model1_1, self.model2_1, self.model3_1, self.model4_1, self.model5_1, self.model6_1] # part affinity fields (2d vecs)
#        modelsS = [self.model1_2, self.model2_2, self.model3_2, self.model4_2, self.model5_2, self.model6_2] # confidence maps
        modelsL = [m for name, m in sorted(self.named_children()) if name.startswith('model') and name.endswith('_1')]
        modelsS = [m for name, m in sorted(self.named_children()) if name.startswith('model') and name.endswith('_2')]
        return modelF, modelsL, modelsS

    def _forward(self, x):
#        T = 2 # of 6
        modelF, modelsL, modelsS = self.getModels()

        # extract features using first 10 layers of VGG-19
        out1Feat = modelF(x)

        # compute several iterations of PAF and confidence map comp.
        tmpFeat = out1Feat
        for t in range(self.T):
            tmpL = modelsL[t](tmpFeat)
            tmpS = modelsS[t](tmpFeat)
            if t != self.T - 1: # don't have to do this in the last iterations
                tmpFeat = torch.cat([tmpL, tmpS, out1Feat], 1)

        return tmpL, tmpS
    
    def _forwardTiming(self, x):
        import time
        T.cuda.synchronize()
        t0 = time.perf_counter()
        y = self._forward(x)
        T.cuda.synchronize()
        t1 = time.perf_counter()
        dt1 = t1 - t0
        print('time elapsed for \'forward\' [ms]: %d' % (dt1*1e3,))
        return y

    def forward(self, x):
        if self.timed:
            return self._forwardTiming(x)
        else:
            return self._forward(x)



