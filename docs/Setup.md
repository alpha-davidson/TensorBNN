---
layout: default
title: Setup
---

# Setup
All python code written here is intended to be used in Python3. The code is dependent upon the packages numpy, emcee, tensorflow, tensorflow-probability, and scipy.

Numpy and scipy can be installed through the command:

```
pip3 install numpy scipy emcee
```

TensorFlow and TensorFlow-probability must be instaled separately. The TensorFlow version should be the most recent (2.3 at the moment). Using a 1.x version will not work, and neither will older versions of 2. It is also highly recomended that this code be run on a gpu due to its high computational complexity. TensorFlow for the gpu can be installed with the command:

```
pip3 install tensorflow-gpu
```

In order to be compatible with this version of tensorflow, the most recent version of tensorflow-probability (0.11) must be installed. This is done with the following command:

```
pip3 install tensorflow-probability
```

In order to use this code you can either clone this repository and copy the Networks folder into a folder named tensorBNN in the main folder of your project, or download it using pip.
```
pip install tensorBNN
git clone https://github.com/alpha-davidson/TensorBNN.git
```

After this, you can use the following command to import the general network obejct, and similar commands for the other objects.
```
from tensorBNN.network import network
```

