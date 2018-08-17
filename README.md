Copyright (c) 2018 ETH Zurich, Lukas Cavigelli


# CBinfer: Change-Based Inference for Convolutional Neural Networks on Video Data 

In this repository, we share a PyTorch-compatible implementation of our work called CBinfer. 
For details, please refer to the papers below. 

If this code proves useful for your research, please cite
> Lukas Cavigelli, Luca Benini,  
"CBinfer: Exploiting Frame-to-Frame Locality for Faster Convolutional Network Inference on Video Streams",  
submitted to IEEE Transactions on Circuits and Systems for Video Technology, 2018.  
DOI (preprint): [10.3929/ethz-b-000282732](https://www.research-collection.ethz.ch/handle/20.500.11850/282732). Available on [arXiv](https://arxiv.org/pdf/1808.05488). 

and/or
> Lukas Cavigelli, Philippe Degen, Luca Benini,  
"CBinfer: Change-Based Inference for Convolutional Neural Networks on Video Data",  
in Proceedings of the 11th International Conference on Distributed Smart Cameras (ICDSC 2017), 
pp. 1-8. ACM, 2017.  
DOI: [10.1145/3131885.3131906](https://doi.org/10.1145/3131885.3131906). Available on [arXiv](https://arxiv.org/abs/1704.04313). 

### Setting up the environment

#### Installing Dependencies
You will need a machine with a CUDA-enabled GPU and the Nvidia SDK installed to compile the CUDA kernels.
Further, we have used conda as a python package manager and exported the environment specifications to `conda-env-cbinfer.yml`. 
You can recreate our environment by running 

```
conda env create -f conda-env-cbinfer.yml -n myCBinferEnv`. 
```
Make sure to activate the environment before running any code. 

#### Compile the CUDA code for PyCBinfer
Switch to the `pycbinfer` folder and run the `./build.sh` script. 
You might want to modify it such that it optimizes for the latest compute capability your GPU supports.
For convenience, these are the build steps run by the script: 

```
nvccflags="-O3 --use_fast_math -std=c++11 -Xcompiler '-fopenmp' --shared --gpu-architecture=compute_52 --compiler-options -fPIC --linker-options --no-undefined"
nvcc -o cbconv2d_cg_backend_$(uname -i).so cbconv2d_cg_backend.cu $nvccflags
nvcc -o cbconv2d_fg_backend_$(uname -i).so cbconv2d_fg_backend.cu $nvccflags

nvccflags="-O3 --use_fast_math -std=c++11 -Xcompiler '-fopenmp' --shared --gpu-architecture=compute_61 --compiler-options -fPIC --linker-options --no-undefined"
nvcc -o cbconv2d_cg_half_backend_$(uname -i).so cbconv2d_cg_half_backend.cu $nvccflags
```

#### Download the scene labeling dataset and model
Switch into the `sceneLabeling` folder and run

```
curl https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/278294/model-sceneLabeling-baseline.tar | tar xz
curl https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/276417/dataset-segm-v2-labeledFrames.tar | tar xz
curl https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/276417/dataset-segm-v2-frameSequences-part1.tar | tar xz
curl https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/276417/dataset-segm-v2-frameSequences-part2.tar | tar xz
```
The data are downloaded and extracted to the `dataset-segmentation` folder, while includes two folder, one for labeled individual frames and one for the unlabeled frame sequences.

#### Download the pose detection dataset
The pose detection dataset consists of several frame sequences from the [CAVIAR dataset](http://homepages.inf.ed.ac.uk/rbf/CAVIAR/) and some Youtube videos. 
Switch into the `poseDetection` folder and run

```
curl https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/278294/model-poseDetection-full.tar | tar xz
curl https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/278294/model-poseDetection-baseline.tar | tar xz
curl https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/278258/dataset-poseDetection.tar | tar xz
```

### Step-by-step Guide
There are two applications to which CBinfer can applied -- scene labeling and pose detection. 
You should chose one to start. The scene labeling evaluations run faster, so this might be the more convenient starting point. 

In a first step, you should convert the baseline (pre-trained, normal) DNN to a CBinfer version using the `modelConverter.py` script. 
At the end you should save the network. There are various ways to convert a network as detailed in the paper, we distinguish them with 
different values for `experimentIdx`. The default value in the script is set to 6, which uses the recursive implementation of CBinfer and converts 
all convolution layers to change-based convolutions and all pooling layers to change-based pooling. After converting the network with this 
script, save the resulting model -- ideally as 'model006.net' to stay in-line with our naming scheme for different experiments. 

Next, you can choose to run an accuracy comparison for a single frame sequence using `run.py`, or a larger set of evaluations using the `eval01.py` script. 


### Software Architecture
The whole setup consists of several parts: 

- The CBinfer implementation of the convolution and max-pooling operations
- Infrastructure common to both applications, such as `tx2power` and `evalTools`
- Application-specific code

#### pycbinfer
PyCBinfer consists of several classes and functions. 

- The conversion function to port a normal DNN model to a CBinfer model as well as some utility functions to reset the model's state can be found in the `__init__.py`. It also includes the function to tune the threshold parameters, which takes many application-specific functions as arguments, such as dataset loader, preprocessor, postprocessor, accuracy evaluator, the model, ...
- The CBConv2d and CBPoolMax2d classes can be found in the `conv2.py` file. They are implementations of a torch.nn.module and include the handling of the tensors, switching between fine-grained and coarse-grained CBinfer, and the overall algorithm flow. 
- The individual processing steps and the lower-level interface to the C/CUDA functions using cffi can be found in `conv2d_cg.py` and `conv2d_fg.py`, respectively. 
- The `cbconv2d_*_backend.cu` contain the CUDA kernels and the C launchers thereof. 

#### evalTools
They contain functions to apply a model to a frame sequence, benchmark performance, measure power, and to filter layers on CBinfer-based models. The benchmark functions are generic and take the following as arguments:

- a model
- a frame set and the number of frames to process
- a preprocessing function to prepare the data
- a postprocessing function to transform the network output to a meaningful result

#### application-specific function for pose detection

- The application-specific code consists of a `modelConverter` script, which load the baseline network, invokes the loading of the dataset, contains all the details of how the network should be converted (which layers, etc.)  and ultimately calls pycbinfer's threshold tuning function. 
- The `poseDetEvaluator` script contains a class with all the metrics and pre-/post-processing functions. 
- The `videoSequenceReader` contains the loader functions for the dataset.
- The `openPose` folder contains all the files to run the OpenPose network and pre-/post-processing until the final extraction of the skeleton. 
- The two evaluation scripts `eval01` and `eval03` contain the code to perform the analysis and visualize the results shown in the paper, e.g. throughput-accuracy trade-offs, power and energy usage, baseline performance measurement, change propagation analyses, etc.

#### tx2power
The `tx2power` file provides some standalone easy-to-use current/voltage/power/energy measurement tools for the Tegra X2 module and the Jetson TX2 board. 
It allows to analyze where the power is dissipated (module: main/all, cpu, ddr, gpu, soc, wifi; board: main/all, 5V0-io-sys, 3v3-sys, 3v3-io-sleep, 1v8-io, 3v3-m.2)
The PowerLogger class provides the means to measure power traces, record events (markers), visualize the data, and obtain key metrics such as total energy. 

### License and Attribution
Please refer to the LICENSE file for the licensing of our code.
For the pose detection application demo, we heavily modified [this](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation) OpenPose implementation. 

