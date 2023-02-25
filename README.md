# UCSDAIProject
AI Project for UCSD

## Project Description
This project uses a Reinforcement Learning Neural Network called DDPG to 
perform automatic analysis of a dataset provided by UCSD. The dataset is comprised of [data description]
and also contains a CNN intended to perform semantic segmentation of the TIFF images
provided by UCSD. The CNN is trained on the dataset and then used to generate an image of the
same resolution that has everything besides the target area blacked out. 
The DDPG neural network is then trained in an environment containing the pre-cut and
post-cut images with the goal being to compute the distance of the axons after the cut.

## Concept Description
### Reinforcement Learning
Reinforcement Learning is a branch of machine learning that teaches a machine what to do
based on an actor-critic system. The actor is the neural network that takes in the state of the
environment and outputs an action. The critic is the neural network that takes in the state of the
environment and outputs a value. The critic is used to train the actor by telling it how good of a
decision it made. The actor is then used to make decisions in the environment. To put this in an analogy:
imagine you are watching an actor play a role in a film. The actor (our neural network) will act, play the role, etc...
The critic will then tell the actor what it could've done better, thereby informing the actor on how to do it better the next time.
The actor does some action (in this case it would be measuring the distance of the cut) and the critic
tells the actor what could've been better (like measure the cut closer to this axon to be more accurate or something like that)

### Convolutional Neural Network/Semantic Segmentation
A Convolutional Neural Network (CNN) is a type of neural network that is used for analyzing images. I won't explain
the process of how a CNN works here, but I will explain how it is used in this project. The CNN is given pairs of images:
one image containing the full area and the other image containing the manually segmented out image that only includes the target area.
The CNN will use these image pairs to take an unsegmented image in the future and then segment it down to
the target area. Here is an example of what that would look like:

![Pre Cut Image](/data/2 DRG/01122023_DRG/Dish 3 WT H2O2/FOV1 5 cuts/overlay/E198_01_12_23_13h02m_57s_ms565__WT_Day4_H2O2.tif?raw=true "Title")

### Environment for Reinforcement Learning
The environment we are using for the Reinforcement Learning is a custom environment that is built using Keras. When I say
"environment" it might sound confusing so to simplify it lets bring back the actor in a film analogy.
The environment is the set that the actor is in. That is all. In this case, the environment is the image of the axons before and after the cut.
The actor is the neural network that is trained to measure the distance of the cut. The critic is the neural network that is trained to tell the actor how
to measure the distance more accurately. This all simply happens within the training environment.


### How the Neural Networks Relate
The CNN is used to generate the images that contain only the target areas that will be used to train the Reinforcement 
Learning Neural Network (DDPG). Therefore, the output of the CNN is the input to the DDPG. The CNN will be used to
segment any future images as well and those segmented images will then have the distance when the axons are cut computed by
the DDPG (reinforcement learning) neural network. 

### How the DDPG Neural Network Works
I won't go into the mathematics of it or the in-depth computer science behind it, but I will just give a general explanantion of how
it works so the code might be better understood. There are two neural networks inside the DDPG neural network. 
One neural network is the actor's neural network: a network designed to perform an action based on an input.
The other neural network is the critic's neural network, which is designed to critique the actor's performance by assigning
it a value based on how well it did (think of this as similar to giving the actor a rating from 1-10).

## How to Run the Code
### Installing Python/Anaconda/Packages (Windows)
Victor you can add your relevant guide here if you would like - Alex

### Installing Python/Anaconda/Packages (M1 Mac)
1. Install Anaconda (https://repo.anaconda.com/archive/Anaconda3-2022.10-MacOSX-arm64.pkg)
2. Open Anaconda Navigator