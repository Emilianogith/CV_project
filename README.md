# CV_project: Pedestrian Intention Estimation on JAAD Dataset
## Abstract

Safe interaction between autonomous vehicles and pedestrians requires accurate estimation of pedestrian intention through visual information analysis. In this study, we explore recent approaches and open challenges in pedestrian intention estimation using computer vision. We provide an overview of techniques, including those based on convolutional neural networks, LSTM, and probabilistic models. Additionally, we investigate challenges related to varying environmental conditions and the presence of partially occluded pedestrians. Furthermore, we propose optimization techniques to enhance the accuracy and reliability of pedestrian intention estimation systems, with a focus on multi-modal data fusion and contextual information integration for a deeper understanding of human behavior in road situations.
Moreover, we examine the utilization of the Pedestrian Intention Estimation JAAD dataset as a standard reference to evaluate the performance of pedestrian intention estimation algorithms. The JAAD dataset comprises a wide range of urban scenes annotated with detailed information on pedestrian intentions, including directional movements, speeds, and predicted behaviors. Its diversity and richness make it a valuable resource for the development and evaluation of computer vision models for pedestrian behavior analysis. Finally, we discuss potential techniques for improving the efficiency of pedestrian intention estimation systems. Among these, the implementation of model compression techniques, such as weight quantization or parameter pruning, could enable more efficient model execution on embedded or real-time devices, making them more suitable for practical implementation in autonomous vehicle systems and driving assistance applications.


## Proposed method 
In this work, we propose a streamlined implementation of a pedestrian intention estimation model. Our approach is inspired by the foundational work presented in [this paper](https://arxiv.org/pdf/2104.05485), but introduces a lighter version capable of efficiently processing a reduced datasets, tailored to operate within the constraints of limited computational resources. To address challenges related to high data volumes, we present a simplified model that leverages hierarchical fusion to handle multi-modal inputs.

The architecture we propose retains the overall structure of similar models, as illustrated below:
![architecture](https://github.com/Emilianogith/CV_project/blob/main/images/hierarchical_fusion.png)

However, unlike the original, our implementation omits the use of global context, focusing instead on the fusion of key inputs for intention estimation.


**keypoints** of the provided architecture:
- After processing each input temporal sequence, an attention mechanism is applied to highlight important sequential features.
- The fusion of inputs from different sources follows a hierarchical structure.
- Finally, after the fusion of visual and non-visual branches, a final attention mechanism is applied, and the extracted features are passed to a fully connected (FC) layer that delivers the final binary prediction.


## How to run the code 

# Dataset Preparation  

Download the [JAAD Annotation](https://github.com/ykotseruba/JAAD) and put `JAAD` file to this project's root directory (as `./JAAD`).  

Download the [JAAD Dataset](http://data.nvision2.eecs.yorku.ca/JAAD_dataset/), and then put the video file `JAAD_clips` into `./JAAD` (as `./JAAD/JAAD_clips`).  

Copy `jaad_data.py` from the corresponding repositories into this project's root directory (as `./jaad_data.py`).  

In order to use the data, first, the video clips should be converted into images. This can be done using script `./JAAD/split_clips_to_frames.sh` following JAAD dataset's instruction.  

Above operation will create a folder called `images` and save the extracted images grouped by corresponding video ids in the `./JAAD/images `folder.  
```
./JAAD/images/video_0001/
				00000.png
				00001.png
				...
./JAAD/images/video_0002/
				00000.png
				00001.png
				...		
...
```
