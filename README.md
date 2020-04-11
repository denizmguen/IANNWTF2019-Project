# IANNWTF2019-Project
This github repo serves as our hand-in for the course "Implementing Artificial Neural Networks with Tensorflow 2019" which was offered at the University of Osnabrück. We attempted to reimplement DDPG (Lillicrap et al. 2015) in tensorflow 2.0.1 from scratch and train the agent on the continuous version of MountainCar-v0. Our implementation is as faithful to the details specified in the 2015 report as we could manage. Most of the steps that we have taken are well documented and a lot of code is additionaly explained through comments on the spot. 


## Guide
There are 3 Notebooks all containing our work with varying levels of completeness.

### Lite.html
   * Open this file, if you just want to have a look at our project without actually running anything. 
   
The next two project files both require 2 (or 3) steps before you can run through them without any complications:
* Clone this repo
* (Create a new environment)
* within the folder, run the following command in your shell: 'pip install -r req.txt'

### Pretrained.ipynb
This notebook is a semi interactive showcase of our project. You just need to open it up and run each individual cell. Pretrained models and results will be loaded and displayed inside the notebook.

### Full.ipynb
Similar to pretrained, you don't have to do anything but run each individual cell. This notebook however, will build and train every model from the ground up. Run this notebook if you wish to reproduce our results. 

