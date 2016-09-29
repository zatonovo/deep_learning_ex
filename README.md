# Deep learning examples
This repository is a supplement to my class Machine Learning and Big Data
taught at the CUNY MS Data Analytics program. More examples will be added
over time.

# Installation
The examples expect a Linux environment with Docker installed. If you meet
this requirement, you can simply use the `Makefile` to build a local Docker
image and run a container. These instructions assume that you've downloaded
the repository to `~/workspace/deep_learning_ex`. If you clone to a different
directory, you will need to change the `Makefile`.

First build the container with `sudo make`. This will download a Docker image
and then install some additional packages. This image will be saved locally
as `zatonovo/dlx`.

Now you are ready to run a container based on the image. You have the option
of starting a bash shell, a Torch session (in Lua), or a Python session.

```
sudo make bash
sudo make torch
sudo make python
```

When you start the session, you'll see that all the files in the host
directory `~/workspace/deep_learning_ex` are visible in the current 
working directory of the image! This is the beauty of volume mapping.
This means that any data and code you create in the container will be
saved in your host file system to use for further analysis. For example,
if you want to visualize your results in R, it's easier to use your
existing R configuration and load the data files, as opposed to running R
from inside the container.

# Examples
## Function approximation
This example approximates a bivariate continuous function in Torch.
For background,
refer to [this primer](https://cartesianfaith.com/2016/09/23/a-primer-on-universal-function-approximation-with-deep-learning-in-torch-and-r/).

To run the example, start a Torch session, as instructed above. You should
see the standard Torch prompt inside Lua.

```
  ______             __   |  Torch7                                         
 /_  __/__  ________/ /   |  Scientific computing for Lua. 
  / / / _ \/ __/ __/ _ \  |  Type ? for help                                
 /_/  \___/_/  \__/_//_/  |  https://github.com/torch         
                          |  http://torch.ch                  
	
th> 
```

From this prompt, type `dofile 'ex_fun_approx.lua'`, which will run the
script. The script itself creates only the most basic of networks. To get
results similar to the end of the primer, you need to add appropriate
[layers](https://github.com/torch/nn/blob/master/doc/simple.md) and [activation functions](https://github.com/torch/nn/blob/master/doc/transfer.md) to the network.
You also need to specify an appropriate
[cost function](https://github.com/torch/nn/blob/master/doc/criterion.md#nn.Criterions) and tune the parameters of the [optimizer](https://github.com/torch/nn/blob/master/doc/training.md#nn.StochasticGradient).
These are all exercises for my students/readers.

# Resources
## Torch
The Torch project has excellent documentation. This example provides a 
[full walkthrough](https://github.com/torch/nn/blob/master/doc/training.md#nn.DoItStochasticGradient) of setting up a neural network and training it.


