# IANNWTF2019-Project
This github repo serves as our hand-in for the course "Implementing Artificial Neural Networks with Tensorflow 2019". We attempted to reimplement DDPG (Lillicrap et al. 2015) in tensorflow 2.0.1 from scratch and train the agent on the continuous version of MountainCar-v0. Our implementation is as faithful to the details specified in the 2015 report as we could manage. Most of the steps that we have taken are well documented and a lot of code is additionaly explained through comments on the spot. Within 100 episodes our agent did NOT learn a policy that solves the problem.


## Guide
There are 3 Notebooks all containing our work with varying levels of completeness.

### Lite.html
   * Open this file, if you just want to have a look at our project without actually running anything. 
   
The next two project files both require 2 (or 3) steps before you can run through them using jupyter without any complications:
* Clone this repo
* (Create a new environment)
* within the folder, run the following command in your shell: 'pip install -r req.txt'

### Pretrained.ipynb
This notebook is a semi interactive showcase of our project. You just need to open it up and run each individual cell. Pretrained models and results will be loaded and displayed inside the notebook.

### Full.ipynb
Similar to pretrained, you don't have to do anything but run each individual cell. This notebook however, will build and train every model from the ground up. Run this notebook if you wish to reproduce our results. 

### ddpg.py
ddpg.py contains all relevant classes and functions in case you prefer to look at the code directly. However, we do not recommend running it blindly.

## Screenshots

### Notebook
<img src="https://github.com/denizmguen/IANNWTF2019-Project/blob/master/img/notebook_screencap.png" height=50% width=50%>
<img src="https://github.com/denizmguen/IANNWTF2019-Project/blob/master/img/notebook_screencap2.png" height=50% width=50%>
<img src="https://github.com/denizmguen/IANNWTF2019-Project/blob/master/img/notebook_screencap3.png" height=50% width=50%>

### Sample Results
<img src="https://github.com/denizmguen/IANNWTF2019-Project/blob/master/results/episodic_rewards/original_ddpg_e100.png" width=65%>

<img src="https://github.com/denizmguen/IANNWTF2019-Project/blob/master/results/episodic_rewards/original_ddpg_test.png" width=65%>

