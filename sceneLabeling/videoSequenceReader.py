#Copyright (c) 2018 ETH Zurich, Lukas Cavigelli
import os
import torch
import skimage.transform
import scipy.misc
import scipy.io
import numpy as np
from functools import lru_cache

@lru_cache(maxsize=2)
def getSequenceNames(validationSetOnly=True):
    """Returns a list of strings identifying the names of available sequences. 
    
    :param validationSetOnly: Choose whether to return only sequences deemed part of the validation set (default: True)
    """
    
    # obtain set of labeled frames
    targetPathFolders = os.listdir('./dataset-segmentation/labeled-frames/')
    seqSet = filter(lambda s: '__' in s, targetPathFolders)
    
    #filter those in validation set
    if validationSetOnly:
        seqSet = filter(lambda s: s.startswith('val_'), seqSet)
        
    #remove 'val_' from sequence name
    def removeVal(s):
        if s.startswith('val_'): 
            s = s[4:]
        return s
    
    #check if corresponding frame sequence exists
    frameSeqFolders = set(os.listdir('./dataset-segmentation/frame-sequences/'))
    def sequenceExists(s):
        #change format
        s = 'volta_18-11-2015-%suhr%s' % (s[0:2], s[2:4])
        return (s in frameSeqFolders)
    
    #get list of potential frame sequence names
    seqSet = filter(sequenceExists, map(removeVal, seqSet))
    return list(seqSet)

@lru_cache(maxsize=100)
def getDataFrames(seqName='1607__11', numFrames=5):
    """Returns two values: A sequence (list) of frames and the target result for the last frame. 
    
    :param seqName: Name of the sequence to load.
    :param numFrames: The number of frames to load before the labeled one. 
    """

    targetPathFolders = os.listdir('./dataset-segmentation/labeled-frames')
    
    # find folder with entire image sequence
    hh = seqName[0:2]
    mm = seqName[2:4]
#    picPath =  '../../scratch/volta_18-11-2015-%suhr%s' % (hh, mm)
    picPath =  './dataset-segmentation/frame-sequences/volta_18-11-2015-%suhr%s' % (hh, mm)

    # load image sequence
    frames = []
    n = int(seqName[6:8])
    try:
        for i in range(numFrames+1):
            imgPath = '%s/%drgb_.png' % (picPath, n - numFrames + i)
            img = scipy.misc.imread(imgPath)
            img = img.astype(np.float)/255
            img = skimage.transform.resize(img, [776, 1040], mode='constant')
            img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()
            frames.append(img)
    except FileNotFoundError:
        return None, None, None

    # find file with ground-truth labels
    folderName = list(filter(lambda s: s.find('%s%s__%s' % (hh, mm, n)) >= 0, targetPathFolders))[0]
    targetPath = './dataset-segmentation/labeled-frames/%s/combined.mat' % folderName
    target = scipy.io.loadmat(targetPath)
    target = torch.from_numpy(target['regionOutput']).long() - 1
        
    return frames, target
