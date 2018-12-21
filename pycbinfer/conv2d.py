#Copyright (c) 2018 ETH Zurich, Lukas Cavigelli
"""
TODO:
 1) matrix multiplication...
 2) make overall test
 3) package
 4) re-benchmark/test
 5) create automatic converter
 6) try more test cases/nets
 7) implement improvements/alternatives
"""

if __name__ == "__main__":   
    from conv2d_fg import cbconvFG
    from conv2d_cg import *
else:
    from .conv2d_fg import cbconvFG
    from .conv2d_cg import *
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class CBPoolMax2d(nn.Module):
    def __init__(self, m):
        super(CBPoolMax2d, self).__init__()

        assert(m.kernel_size == (2,2) and m.stride == (2,2))
        self.stride = m.stride
        self.kernel_size = m.kernel_size
        self.ceil_mode = m.ceil_mode
        self.propChangeIndexes = False
        self.register_buffer('outputState', torch.FloatTensor(0))
        
        self.clearMemory()

    def clearMemory(self):
        if not(hasattr(self, 'outputState')) or ('outputState' not in self._buffers):
            self.register_buffer('outputState', torch.FloatTensor(0))
        
        self.outputState = self.outputState.new(0)

    def getStateTensors(self):
        state = []
        if hasattr(self, 'outputState'):
            state += [self.outputState]
        return state
        
    def forward(self, inp):
        assert(type(inp) == tuple and inp[0] == 'changeIndexes')
        input = inp[1].data.contiguous()
        changeIndexes = inp[2].data.contiguous()
        if len(changeIndexes) != 0:
            assert(changeIndexes.dim() == 1)
            
            nc, h, w = input.size(-3), input.size(-2), input.size(-1)
            if self.ceil_mode:
                oh, ow = (h-1)//2+1, (w-1)//2+1
            else:
                oh, ow = h//2, w//2
            if list(self.outputState.size()[-3:]) != [nc, oh, ow]:
                self.outputState.resize_(1, nc, oh, ow).fill_(1e1000)
            assert(self.outputState.is_cuda)
            
            maxPool2d(input, self.outputState, changeIndexes, self.kernel_size, self.stride,
                      useHalf=(type(input) == torch.cuda.HalfTensor))
            
#            output = torch.nn.functional.max_pool2d(torch.nn.functional.Variable(input), 
#                                                    self.kernel_size, stride=self.stride, 
#                                                    ceil_mode=self.ceil_mode).data
#            self.outputState = output
        
        output = self.outputState.clone()
        
        if self.propChangeIndexes:
            return 'changeIndexes', F.Variable(output), F.Variable(changeIndexes)
        else:
            return F.Variable(output)

    def __repr__(self):
        s = ('{name} (k={kernel_size}, s={stride}, ceil_mode={ceil_mode}')
        s += ', propChgIdxs={propChangeIndexes}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class CBConv2d(nn.Module):
    def __init__(self, m, threshold):
        super(CBConv2d, self).__init__()

        assert(m.groups == 1 and m.transposed == False)
        assert(m.output_padding == (0,0) and m.padding == (m.kernel_size[-2]//2,m.kernel_size[-1]//2))
        assert(m.dilation == (1,1) and m.stride == (1,1))
        self.groups = m.groups
        self.transposed = m.transposed
        self.output_padding = m.output_padding
        self.padding = m.padding
        self.dilation = m.dilation
        self.stride = m.stride
        self.kernel_size = m.kernel_size
        self.in_channels = m.in_channels
        self.out_channels = m.out_channels

        assert(m.weight is not None)# and m.bias is not None)
        self.weight = m.weight
        self.bias = m.bias

        self.threshold = threshold

        self.clearMemory()
        
        self.withReLU = False
        self.saveChangeMap = False
        self.propChangeIndexes = False
        self.gatherComputationStats = False
        self.finegrained = False
        self.copyInput = True
        self.feedbackLoop = False

#        self.changeMap = torch.ByteTensor() # ONLY VOLATILE!!!
#        self.changeIndexes = torch.LongTensor() # ONLY VOLATILE!!!
#        Xmatrix, Ymatrix = ... IS ONLY TEMP/volatile
#        prevInput, prevOutput... set link at run-time

    def clearMemory(self):
        if not(hasattr(self, 'prevInput')):
            self.register_buffer('prevInput', self.weight.data.new(0))
        elif 'prevInput' not in self._buffers:
            # if attribute exists, but is not a properly registered buffer, fix it
            tmp = self.prevInput
            del self.prevInput
            self.register_buffer('prevInput', tmp)
        if type(self.prevInput) != type(self.weight.data):
            self.prevInput = self.prevInput.type_as(self.weight.data)
            
            
        if not(hasattr(self, 'prevOutput')):
            self.register_buffer('prevOutput', self.weight.data.new(0))
        elif 'prevOutput' not in self._buffers:
            tmp = self.prevOutput
            del self.prevOutput
            self.register_buffer('prevOutput', tmp)
        if type(self.prevOutput) != type(self.prevInput):
            self.prevOutput = self.prevOutput.type_as(self.prevInput)            
            
        self.prevInput.fill_(-float('inf')).resize_(0) 
        self.prevOutput.fill_(-float('inf')).resize_(0)
#        self.prevInput.fill_(-1e100).resize_(0) 
#        self.prevOutput.fill_(-1e100).resize_(0)
        if hasattr(self, 'compStats'):
            self.compStats = None
            
    def getStateTensors(self):
        state = []
        if hasattr(self, 'prevInput'):
            state += [self.prevInput]
        if hasattr(self, 'prevOutput'):
            state += [self.prevOutput]
        return state

    def forward_fg(self, inp):
        input = inp.data
        
        if self.prevInput.size() != input.size():
            #init prevOutput
            self.prevOutput = F.conv2d(F.Variable(input), self.weight, 
                                     padding=tuple(s//2 for s in self.weight.size()[2:]),
                                     bias=self.bias).data
        else:
            po = self.prevOutput.clone()
            self.prevOutput = cbconvFG(input, self.prevInput, po, self.weight, self.threshold)
            
        outp = F.Variable(self.prevOutput)
        if self.withReLU:
            outp = F.relu(outp)
        self.prevInput = input
        return outp

    def forward_normal(self, inp):
        #input parsing and checks
        if type(inp) == tuple:
            if inp[0] == 'changeIndexes':
                input = inp[1].data.contiguous()
                changeIndexes = inp[2].data.contiguous()
                assert(changeIndexes.dim() == 1)
            else:
                assert(False)
        else:
            input = inp.data.contiguous()
        assert(input.size(-3) == self.in_channels)
        assert(input.dim() == 4)
        
        if self.prevInput.size() != input.size():
#            self.prevInput = input.new(input.size()).fill_(1e1000)
            self.prevInput.resize_as_(input).fill_(1e1000)
        outpSize = list(input.size())
        outpSize[-3] = self.out_channels
        outpSize = torch.Size(outpSize)
        if self.prevOutput.size() != outpSize:
            self.prevOutput.resize_(outpSize).fill_(1e1000)
        
        if self.gatherComputationStats:
            #gather statistics based on fine-grained evaluation per input feature map
            #additional opt. would be to act on input pixel instead of affected output pixels
            changeTensor = (input - self.prevInput).abs().gt(self.threshold)
            proped = torch.nn.functional.conv2d(
                            changeTensor.float(), 
                            changeTensor.new(changeTensor.size(-3),1,self.weight.size(2),self.weight.size(3)).fill_(1).float(), 
                            groups=changeTensor.size(-3)).data.gt(0)
#            proped = torch.nn.functional.conv2d(
#                            torch.nn.functional.Variable(changeTensor.float()), 
#                            torch.nn.functional.Variable(changeTensor.new(
#                                    changeTensor.size(-3),1,self.weight.size(2),self.weight.size(3)).fill_(1).float()), 
#                            groups=changeTensor.size(-3)).data.gt(0)
            opsPerValue = self.weight.size(0)*self.weight.size(2)*self.weight.size(3)*2
            compStats = dict(
                numInputChangesPerFeatureMap = changeTensor.sum()*opsPerValue, # no. of Ops with FG FM&SP inference
                numInputChanges = changeTensor.sum(-3).gt(0).sum()*changeTensor.size(-3)*opsPerValue, # no. of Ops w/ FG SP
                numInputPropedChangesPerFeatureMap = proped.sum()*opsPerValue, # no. of Ops w/ FG FM
                numInputPropedChanges = proped.sum(-3).gt(0).sum()*changeTensor.size(-3)*opsPerValue, # no. of Ops w/ CG
                totalInputValues = changeTensor.size(-1)*changeTensor.size(-2)*changeTensor.size(-3)*opsPerValue # no. of Ops cuDNN
            )
            self.compStats = compStats

        if 'changeIndexes' not in locals():
            if input.is_cuda:
                changeMap = changeDetection(input, self.prevInput, self.kernel_size, 
                                            self.threshold, updateInputState=self.feedbackLoop, 
                                            useHalf=(type(input) == torch.cuda.HalfTensor))
            else:
                assert(self.feedbackLoop == False)
                changeMap = changeDetection_python(input, self.prevInput, self.kernel_size, self.threshold)
                
            if self.saveChangeMap:
                self.changeMap = changeMap
            
            changeIndexes = changeIndexesExtr_python(changeMap)
            
        if not(self.feedbackLoop):
            if self.copyInput:
                self.prevInput.copy_(input)
            else:
                self.prevInput = input
                
        if changeIndexes.numel() != 0:
            if self.prevInput.is_cuda:
                Xmatrix = genXMatrix(self.prevInput, changeIndexes, self.kernel_size, 
                                     useHalf=(type(self.prevInput) == torch.cuda.HalfTensor))
            else:
                Xmatrix = genXMatrix_python(self.prevInput, changeIndexes, self.kernel_size)
            if self.bias is None: 
                Ymatrix = matrixMult_python(Xmatrix, self.weight.data, None)
            else:
                Ymatrix = matrixMult_python(Xmatrix, self.weight.data, self.bias.data)
            Ymatrix = Ymatrix.transpose(0,1)
            if self.prevInput.is_cuda:
                output = updateOutput(Ymatrix, changeIndexes, self.prevOutput, 
                                      withReLU=self.withReLU, 
                                      useHalf=(type(input) == torch.cuda.HalfTensor))
            else:
                output = updateOutput_python(Ymatrix, changeIndexes, self.prevOutput, withReLU=self.withReLU)
            self.prevOutput = output # not needed, since internally prevOutput is updated and output = self.prevOutput
    
        if self.propChangeIndexes:
            return 'changeIndexes', self.prevOutput, changeIndexes
#            return 'changeIndexes', F.Variable(self.prevOutput), F.Variable(changeIndexes)
        else:
            return self.prevOutput
#            return F.Variable(self.prevOutput) 
        
    def forward(self, inp):
        
        self._setDefaultValues()

        if self.finegrained:
            assert(self.feedbackLoop == False)
            return self.forward_fg(inp)
        else:
            return self.forward_normal(inp)

    def __repr__(self):
        self._setDefaultValues()
            
        s = ('{name} (th={threshold}, {in_channels}->{out_channels}, k={kernel_size}'
             ', s={stride}, copyInput={copyInput}')
        if self.padding != (0,) * len(self.padding):
            s += ', pad={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', outpad={output_padding}'
        if self.groups != 1:
            s += ', grp={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.withReLU:
            s += ', withReLU={withReLU}'
        s += ', propChgIdxs={propChangeIndexes}'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def _setDefaultValues(self):
        if not(hasattr(self, 'saveChangeMap')):
            self.saveChangeMap = False
        if not(hasattr(self, 'propChangeIndexes')):
            self.propChangeIndexes = False
        if not(hasattr(self, 'gatherComputationStats')):
            self.gatherComputationStats = False
        if not(hasattr(self, 'finegrained')):
            self.finegrained = False
        if not(hasattr(self, 'copyInput')):
            self.copyInput = True
        if not(hasattr(self, 'feedbackLoop')):
            self.feedbackLoop = False

#if __name__ == "__main__":   
#    passed = cbconvFG_test1(gpu=False)
#    if passed:
#        print('cpu: ok')
#    else:
#        print('cpu: FAIL')
#        assert(False) 
#    passed = cbconvFG_test1(gpu=True)
#    if passed:
#        print('gpu: ok')
#    else:
#        assert(False)
    
#    ffi.dlclose(C)
#
