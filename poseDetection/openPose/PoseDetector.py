#Copyright (c) 2018 ETH Zurich, Lukas Cavigelli
import cv2
import torch
import torch as T
import torch.nn as nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms
import numpy as np
from . import util
from .config_reader import config_reader
from .PoseModel import PoseModel
from scipy.ndimage.filters import gaussian_filter
import math
import matplotlib.pyplot as plt


class PoseDetector:
    def __init__(self, poseModel=None):
        # find connection in the specified sequence, center 29 is in the position 15
        self.limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
                        [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
                        [1,16], [16,18], [3,17], [6,18]]
        
        # the middle joints heatmap correpondence
        self.mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], \
                       [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], \
                       [55,56], [37,38], [45,46]]
        
        # visualization colors (joints and limbs)
        self.colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                       [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                       [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

        self.param_, self.model_ = config_reader()
        
        if poseModel is None:
            self.model = PoseModel.fromFile()#(weightFilePath='./model/pose_model.pth', T=2)
        else:
            self.model = poseModel
            
        self.enableHeatMapSmoothing = True # if False, can save ~0.9s of 4.0s
        self.enableInpainting = True # if False, can save 0.7s of 4.0s
            
    def preprocess(self, oriImg): # ~13ms
        # preprocessing (scaling, padding, dim.-order)
#    
#        outputUpdate_kernel(PtrHolder(YMatrix.contiguous()),
#                            PtrHolder(prevOutput.contiguous()),
#                            PtrHolder(changeIndexes.contiguous()),
#                            np.int32(outW*outH),
#                            np.int32(numChanges),
#                            np.int32(outC),
#                            np.byte(withReLU),
#                            block=(CUDA_NUM_THREADS,1,1), grid=((numChanges*outC-1)//CUDA_NUM_THREADS + 1,1))
#        scale = self.model_['boxsize'] / float(oriImg.shape[0])
#        boxsize = (int(oriImg.shape[1]*scale), int(oriImg.shape[0]*scale)) # for pytorch below 0.3
        scale = self.model_['boxsize'] / float(oriImg.shape[0])
        boxsize = (int(oriImg.shape[0]*scale), int(oriImg.shape[1]*scale)) #for pytorch 0.3 and up...
    
#        oriImgTensor = torch.from_numpy(oriImg).transpose(2,1).transpose(0,1).unsqueeze(0).float().contiguous()
#        #plt.imshow(oriImgTensor.data[0].transpose(0,1).transpose(2,1).contiguous().numpy())
#        imageToTest = torch.nn.functional.upsample(torch.nn.functional.Variable(oriImgTensor)/255.0, boxsize, mode='nearest').data*255.0
#        imageToTest = imageToTest[0]
#        plt.imshow(imageToTest.transpose(0,1).transpose(2,1).contiguous().numpy())
        
        imageToTest = torchvision.transforms.ToPILImage()(T.from_numpy(oriImg).transpose(1,2).transpose(0,1))
        imageToTest = torchvision.transforms.Scale(boxsize, interpolation=3)(imageToTest)
        imageToTest = torchvision.transforms.ToTensor()(imageToTest)
        
        imageToTest_padded = util.padBottomRight(imageToTest, self.model_['stride'], self.model_['padValue']/256.0)
        imageToTest_padded = imageToTest_padded.mul_(255.0/256.0).add_(-0.5) # yes, really...
        return imageToTest_padded
        
        
    def detectAndInpaint(self, oriImg, gpu=True):
    
        canvas = np.copy(oriImg) # for visualization background
        imageToTest_padded = self.preprocess(oriImg)
        feed = Variable(imageToTest_padded.unsqueeze(0))
        if gpu: 
            feed = feed.cuda()
    
        # feed the NN model
        paf,heatmap = self.model(feed)
    
        # rescale again to input resolution
        heatmap = nn.functional.upsample(heatmap, size=(oriImg.shape[0], oriImg.shape[1]), mode='bilinear').data
        paf = nn.functional.upsample(paf, size=(oriImg.shape[0], oriImg.shape[1]), mode='bilinear').data
        
        # find heatmap maxima 2
        #generate peak map based on local maxima on smoothed input data
        hm = heatmap[0].cpu().numpy()
        if self.enableHeatMapSmoothing:
            hmSmooth = hm.copy()
            for c in range(hm.shape[0]):
                hmSmooth[c] = gaussian_filter(hm[c], sigma=3)
            hmSmooth = torch.from_numpy(hmSmooth)
        else:
            hmSmooth = heatmap[0].cpu()
            
#        hmSmooth = hmSmooth.cuda()
        peakMap = ((hmSmooth[:,1:-2,1:-2] >= hmSmooth[:,2:-1,1:-2]) + #and #x[t] >= x[t+1]
                       (hmSmooth[:,1:-2,1:-2] >= hmSmooth[:,0:-3,1:-2]) + #and #x[t] >= x[t-1]
                       (hmSmooth[:,1:-2,1:-2] >= hmSmooth[:,1:-2,2:-1]) + #and 
                       (hmSmooth[:,1:-2,1:-2] >= hmSmooth[:,1:-2,0:-3]) + #and
                       (hmSmooth[:,1:-2,1:-2] >= self.param_['thre1'])) == 5
#        peakMap = peakMap.cpu() # small speed-up on GPU: 0.12s of 4.0s
        #NMS is being added to torchvision...
        
        # extract coordinates, value, index
        all_peaks = []
        numPeaks = 0
        for partIdx in range(peakMap.size(0)):
            partPeaks = [(p[1]+1, p[0]+1, hm[partIdx,p[0]+1,p[1]+1], numPeaks + idx) for idx, p in enumerate(peakMap[partIdx].nonzero())]
            all_peaks.append(partPeaks)
            numPeaks += len(partPeaks)
          
    
    
        # find connections between detected joints
        connection_all = []
        special_k = []
        mid_num = 10
    
        #for each limb, find candidate points (local maxima) of the two joints it is connected two
        epsilon1 = 1e-10
        pafCpu = paf[0].cpu().numpy()
        for k in range(len(self.mapIdx)):
            score_mid = pafCpu[[x-19 for x in self.mapIdx[k]],:,:]
            candsA = all_peaks[self.limbSeq[k][0]-1]
            candsB = all_peaks[self.limbSeq[k][1]-1]
            indexA, indexB = self.limbSeq[k]
            if(len(candsA) != 0 and len(candsB) != 0):
                connection_candidate = []
                for i, candA in enumerate(candsA):#range(nA):
                    for j, candB in enumerate(candsB):#range(nB):
                        vec = np.subtract(candsB[j][:2], candsA[i][:2])
                        norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                        vec = np.divide(vec, norm)
    
                        pointsOnLimb = list(zip(np.linspace(candsA[i][0], candsB[j][0], num=mid_num), 
                                            np.linspace(candsA[i][1], candsB[j][1], num=mid_num)))
    
                        vec_x = np.array([score_mid[0, 
                                                    int(round(pointsOnLimb[I][1])), 
                                                    int(round(pointsOnLimb[I][0]))] 
                                          for I in range(len(pointsOnLimb))])
                        vec_y = np.array([score_mid[1, 
                                                    int(round(pointsOnLimb[I][1])), 
                                                    int(round(pointsOnLimb[I][0]))] 
                                          for I in range(len(pointsOnLimb))])
    
                        score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
#                        print('norm: %d' % (norm,))
                        score_with_dist_prior = score_midpts.mean() + min(0.5*oriImg.shape[0]/(norm+epsilon1)-1, 0)
                        criterion1 = len(np.nonzero(score_midpts > self.param_['thre2'])[0]) > 0.8 * len(score_midpts)
                        criterion2 = score_with_dist_prior > 0
                        if criterion1 and criterion2:
                            connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candsA[i][2]+candsB[j][2]])
    
                #sort connection candidates by score and iteratively connected them (if none of the joints has been connect before)
                connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True) 
                connection = np.zeros((0,5))
                for c in range(len(connection_candidate)):
                    i,j,s,_ = connection_candidate[c]
                    if(i not in connection[:,3] and j not in connection[:,4]):
                        connection = np.vstack([connection, [candsA[i][3], candsB[j][3], s, i, j]])
                        if(len(connection) >= min(len(candsA), len(candsB))):
                            break
    
                connection_all.append(connection)
            else:
    
                special_k.append(k)
                connection_all.append([])
    
    
        # last number in each row is the total parts number of that person
        # the second last number in each row is the score of the overall configuration
        subset = -1 * np.ones((0, 20))
        candidate = np.array([item for sublist in all_peaks for item in sublist])
    
        for k in range(len(self.mapIdx)):
            if k not in special_k:
                partAs = connection_all[k][:,0]
                partBs = connection_all[k][:,1]
                indexA, indexB = np.array(self.limbSeq[k]) - 1
    
                for i in range(len(connection_all[k])): #= 1:size(temp,1)
                    found = 0
                    subset_idx = [-1, -1]
                    for j in range(len(subset)): #1:size(subset,1):
                        if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                            subset_idx[found] = j
                            found += 1
    
                    if found == 1:
                        j = subset_idx[0]
                        if(subset[j][indexB] != partBs[i]):
                            subset[j][indexB] = partBs[i]
                            subset[j][-1] += 1
                            subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                    elif found == 2: # if found 2 and disjoint, merge them
                        j1, j2 = subset_idx
                        membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                        if len(np.nonzero(membership == 2)[0]) == 0: #merge
                            subset[j1][:-2] += (subset[j2][:-2] + 1)
                            subset[j1][-2:] += subset[j2][-2:]
                            subset[j1][-2] += connection_all[k][i][2]
                            subset = np.delete(subset, j2, 0)
                        else: # as like found == 1
                            subset[j1][indexB] = partBs[i]
                            subset[j1][-1] += 1
                            subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
    
                    # if find no partA in the subset, create a new subset
                    elif not found and k < 17:
                        row = -1 * np.ones(20)
                        row[indexA] = partAs[i]
                        row[indexB] = partBs[i]
                        row[-1] = 2
                        row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                        subset = np.vstack([subset, row])
    
        # delete some rows of subset which has few parts occur
        deleteIdx = []
        for i in range(len(subset)):
            if subset[i][-1] < 4 or subset[i][-2]/subset[i][-1] < 0.4:
                deleteIdx.append(i)
        subset = np.delete(subset, deleteIdx, axis=0)
    
        # inpaint joints
        if self.enableInpainting:
            for jointTypeIdx, jointColor in zip(range(18), self.colors):
                detectedJoints = all_peaks[jointTypeIdx]
                for joint in detectedJoints:
                    cv2.circle(canvas, joint[0:2], 4, jointColor, thickness=-1)
    
        # inpaint limbs
        if self.enableInpainting:
            stickwidth = 4
            for limb, limbColor in zip(self.limbSeq, self.colors):
                for subs in subset:
                    index = subs[np.array(limb)-1]
                    if -1 in index:
                        continue
                    cur_canvas = canvas.copy()
                    Y = candidate[index.astype(int), 0]
                    X = candidate[index.astype(int), 1]
                    mX = np.mean(X)
                    mY = np.mean(Y)
                    length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
                    angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
                    polygon = cv2.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
                    cv2.fillConvexPoly(cur_canvas, polygon, limbColor)
                    canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    
        return canvas

