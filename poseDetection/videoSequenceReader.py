#Copyright (c) 2018 ETH Zurich, Lukas Cavigelli
import os
import cv2
from functools import lru_cache

#basepath = './dataset-poseDetection/sequenceSets/'
#seqName = '2DiQUX11YaY-720p-seq000-twoDancers'
#seqName = 'ski-seq000'
#numFrames = 5

@lru_cache(maxsize=32)
def getSequenceNames(validationSetOnly=True):
    assert(validationSetOnly)
    datasetDir = './dataset-poseDetection/sequenceSets/' # folder with images for which there are labels
    labeledImagePaths = next(os.walk(datasetDir))[1]#list(filter(lambda s: s.find('.') < 0, os.listdir(datasetDir)))
    return labeledImagePaths


#@lru_cache(maxsize=100)
def getDataFrames(seqName='ski-seq000', numFrames=5):
    datasetDir = './dataset-poseDetection/sequenceSets/' # folder with images for which there are labels
    seqDir = datasetDir + seqName + '/'
    # sort by name to get sequence right; skip hidden files: 
    frameFiles = sorted(filter(lambda s: not(s.startswith('.')), os.listdir(seqDir))) 
    if numFrames is not None and numFrames >= 0:
        frameFiles = frameFiles[:numFrames]
    #read frames
    frames = []
    for frameFile in frameFiles:
        framePath = seqDir + frameFile
        frame = cv2.imread(framePath)
        frames.append(frame)
        
    target = None
    return frames, target


#class VideoSequenceReader:
#    @staticmethod
#    def getSequenceNames(validationSetOnly):
#        """returns a list of names/identifiers of available sequences"""
#        raise NotImplementedError()
#    @staticmethod
#    def getDataFrames(seqName, numFrames):
#        """returns the frames of the sequence and (optionally, else None) the target output/ground truth"""
#        raise NotImplementedError()
    
#class SceneLabelingVideoSequenceReader(VideoSequenceReader):
#    @staticmethod
#    def getSequenceNames(validationSetOnly=True):
#        return getSequenceNames(validationSetOnly)
#    
#    @staticmethod
#    def getDataFrames(seqName='ski-seq000', numFrames=5):
#        return getDataFrames(seqName, numFrames)

#    @staticmethod
#    @lru_cache(maxsize=32)
#    def getSequenceNames(validationSetOnly=True):
#        assert(validationSetOnly)
#        datasetDir = './dataset/sequenceSets/' # folder with images for which there are labels
#        labeledImagePaths = next(os.walk(datasetDir))[1]#list(filter(lambda s: s.find('.') < 0, os.listdir(datasetDir)))
#        return labeledImagePaths    
#    
#    @staticmethod
#    @lru_cache(maxsize=100)
#    def getDataFrames(seqName='ski-seq000', numFrames=5):
#        datasetDir = './dataset/sequenceSets/' # folder with images for which there are labels
#        seqDir = datasetDir + seqName + '/'
#        # sort by name to get sequence right; skip hidden files: 
#        frameFiles = sorted(filter(lambda s: not(s.startswith('.')), os.listdir(seqDir))) 
#        frameFiles = frameFiles[:numFrames]
#        #read frames
#        frames = []
#        for frameFile in frameFiles:
#            framePath = seqDir + frameFile
#            frame = cv2.imread(framePath)
#            frames.append(frame)
#            
#        target = None
#        return frames, target

