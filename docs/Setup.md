---
layout: default
title: Setup
---

# Setup
All python code written here is intended to be used in Python3. The code is dependent upon the packages numpy, tensorflow, tensorflow-probability, and scipy.

Numpy and scipy can be installed through the command:

```
pip3 install numpy scipy
```

The tensorflow version must be 2.0. Using a 1.x version will not work. It is also highly recomended that this code be run on a gpu due to its high computational complexit. Tensorflow 2.0 for the gpu can be installed with the command:

```
pip3 install tensorflow-gpu==2.0.0-beta1
```

In order to be compatible with tensorflow 2.0, the nightly version of tensorflow-probability must be installed. This is done with the following command:

```
pip3 install tfp-nightly
```

In order to use this code simply clone this repository and copy the Networks folder into the main folder of your project.
```
git clone https://github.com/alpha-davidson/TensorBNN.git
```

After this, you can use the following command to import the general network obejct, and similar commands for the other objects.
```
from Networks.network import network
```

