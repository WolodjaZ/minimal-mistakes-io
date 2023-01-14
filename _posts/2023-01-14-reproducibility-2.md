---
title: "Reproducibility in Tensorflow/PyTorch/JAX Part 2/2"
excerpt: "This is the second part of the blog about reproducibility in TensorFlow, PyTorch, and JAX."
data: 2023-01-14
languages: [python, jax, tensorflow, pytorch]
tags: [Reproducibility, JAX, Tensorflow, PyTorch]
toc: true
---


Welcome back to the second part of this article on reproducibility in TensorFlow, PyTorch, and JAX. In the first part, we discussed the importance of reproducibility in machine learning and the factors that can impact reproducibility in these frameworks.

To achieve reproducibility in machine learning, it is important to carefully control the sources of randomness in your model as well as the environment in which your code is executed. This includes setting up a consistent environment with defined package versions and making necessary adjustments to environment variables used by certain libraries.

## Environment reproducibility

In this section, we will focus on creating a consistent environment for our machine learning project. A good first step is to create a virtual environment, which allows us to separate the dependencies for this project from other projects and ensure that we are using the same package versions across different systems. This is especially useful when working on multiple projects concurrently, or when using different package versions of the same library. Let's get started.


```python
# The name of your env can be `poop` etc as you wish but it is very common to give venv
# and some lib use this name for detecting env so you understand.
% python3 -m venv ./venv && source ./venv/bin/activate
```

Now that we have prepared our environment and activated it, we can proceed with installing the necessary packages. One of the tools we will be using in this project is [MLCube](https://mlcommons.github.io/mlcube/), which we discussed in the first part of this article. To install MLCube and its dependencies, we can use the following command:


```python
# Install MLCube and MLCube docker/singularity runners
% pip install mlcube mlcube-docker mlcube-singularity
# Show platform configurations. A platform is a configured instance of a runner.
% mlcube config --get platforms
```

This will install MLCube and all the necessary packages required to use it. With MLCube installed, we are now ready to move on to the next step of setting up our reproducible machine learning environment.

We will install the necessary packages for our code using containers such as [Docker](https://www.docker.com) and [Singularity](https://singularity-userdoc.readthedocs.io/en/latest/) to create an environment for our code. To do this, we need to create a file called qrequirements.txtq that lists the packages and their specific versions that we need for the project. In my project, the `requirements.txt` file looks like this:
```text
flax==0.6.3
ipython==8.7.0
jax==0.3.25
jaxlib==0.3.25
optax==0.1.4
Pillow==9.3.0
tensorflow==2.11.0
tensorflow-datasets==4.7.0
torch==1.13.0
torchvision==0.14.0
hydra-core==1.3.1
ml-collections==0.1.1
```
For our project, the packages we will be using include Tensorflow, PyTorch, and JAX, as well as some additional packages like [Flax](https://flax.readthedocs.io) and [Optax](https://optax.readthedocs.io/en/latest/) for JAX. If you are familiar with these languages, you should recognize these packages as being common used in their respective frameworks. In addition to these packages, we will also be using [Hydra](https://hydra.cc), a package that allows us to run code based on `YAML` files. This can be useful if we need to change certain settings in our code, as it allows us to store them in `YAML` files rather than hardcoding them into the code.

After we have created the `requirements.txt` file, we need to create a `Dockerfile` image, where we will define everything needed for our container:
```text
# Base image
FROM python:3.9-slim
LABEL author="Vladimir Zaigrajew"

# Update software
RUN apt-get update

# Copy requirements and install
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy script files
COPY mnist_jax.py /workspace/mnist_jax.py
COPY mnist_pytorch.py /workspace/mnist_pytorch.py
COPY mnist_tensorflow.py /workspace/mnist_tensorflow.py
COPY run.sh /workspace/run.sh

# Copy config files
COPY mlcube.yaml /mlcube/mlcube.yaml
COPY workspace/train.yaml /mlcube/workspace/train.yaml

# Make run.sh executable
RUN chmod +x /workspace/run.sh

# Run run.sh
ENTRYPOINT ["/bin/bash", "/workspace/run.sh"]
```
and Singularity recipe `Singularity.recipe`:
```text
# Base image from docker
BootStrap: docker
FROM python:3.9-slim

%labels
    Maintainer Vladimir Zaigrajew

%post
    # Update software
    apt-get update

    # Install requirements.txt
    pip install --no-cache-dir -r /requirements.txt

    # Make run.sh executable
    chmod +x /workspace/run.sh

# Copy necessary files
%files
    requirements.txt /requirements.txt
    run.sh /workspace/run.sh
    mnist_jax.py /workspace/mnist_jax.py
    mnist_pytorch.py /workspace/mnist_pytorch.py
    mnist_tensorflow.py /workspace/mnist_tensorflow.py

    mlcube.yaml /mlcube/mlcube.yaml
    workspace/train.yaml /mlcube/workspace/train.yaml

# Run run.sh
%runscript
    /bin/bash /workspace/run.sh "$@"
```
The last thing we need to define is our MLCube contract `mlcube.yaml`:
```yaml
# Name of this MLCube.
name: repro
# Brief description for this MLCube.
description: Reproducibility in Tensorflow/PyTorch/JAX
# List of authors/developers. 
authors:
  - {name: "Vladimir Zaigrajew", email: "vladimirzaigrajew@gmail.com"}

# Platform description. This is where users can specify MLCube resource requirements, such as 
# number of accelerators, memory and disk requirements etc. The exact structure and intended 
# usage of information in this section is work in progress. This section is optional now.
platform:
  accelerator_count: 1
  accelerator_maker: NVIDIA
  accelerator_model: GeForce RTX 2080 Ti - 11MiB
  CUDA_version: 11.4
  need_internet_access: True

# Configuration for docker runner (additional options can be configured in system settings file).
docker:
  image: wolodja55/repro:0.0.1
  image: wolodja55/repro:0.0.1
  build_context: .
  gpu_args: '--gpus=all'
  build_file: DockerBuildfile
  build_strategy: auto

# Configuration for singularity runner (additional options can be configured in system settings 
# file).
singularity:
  image: repro-0.0.1.sif

# Section where MLCube tasks are defined.
tasks:
  pytorch:
    parameters:
        inputs:
          workspace: {type: directory, default: ""}
    # `pytorch` task. It has one input
  tensorflow:
    parameters:
        inputs:
          workspace: {type: directory, default: ""}
    # `tensorflow` task. It has one input
  jax:
    parameters:
        inputs:
          workspace: {type: directory, default: ""}
    # `jax` task. It has one input
```
We have all the necessary components for setting up our environment. You can see the structure of my environment that I used to obtain these results. I will also test the environment on different machines with different GPUs and CPUs to ensure that it is reproducible.

## Well, after all the wait, perhaps it's time to code 👩‍💻

To illustrate how to set up reproducible training using different frameworks, let's consider the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset as an example. In the following sections, we'll show you examples of reproducible training in Pytorch, Tensorflow, and JAX, and provide a brief summary of each example. Next, based on those examples I will create files for our pipeline and test it, so let's start.

First, we will set some variables. We will only set `CUDA_VISIBLE_DEVICES` to ensure that we are working on one device and `CUBLAS_WORKSPACE_CONFIG`. `CUBLAS_WORKSPACE_CONFIG` may be needed by Pytorch becouse, bit-wise reproducibility is not guaranteed across toolkit versions because the implementation might differ due to some implementation changes. You will get to choose `:16:8` (may limit overall performance) or `:4096:8` (will increase library footprint in GPU memory by approximately 24MiB). Any of those settings will allow for deterministic behavior even with multiple concurrent streams sharing a single cuBLAS handle.

To ensure reproducibility in our deep learning code, we need to set some variables that will control the behavior of our model. Specifically, we will set the `CUDA_VISIBLE_DEVICES` variable to ensure that we are using a specific GPU device, and the `CUBLAS_WORKSPACE_CONFIG` variable to configure cuBLAS, as bit-wise reproducibility is not guaranteed across toolkit versions.

The `CUDA_VISIBLE_DEVICES` variable is used to specify which GPU devices are visible to the code. By setting this variable, we can ensure that our code is only using a single GPU device. The `CUBLAS_WORKSPACE_CONFIG` variable can take two values: `:16:8` and `:4096:8`. The `:16:8` setting may limit overall performance, but it allows for deterministic behavior even with multiple concurrent streams sharing a single cuBLAS handle. The `:4096:8` setting will increase the library footprint in GPU memory by approximately 24MiB, but it also allows for deterministic behavior.


```python
# set gpu id to test it on one device
%env CUDA_VISIBLE_DEVICES=0
# You may need to set this variables by PyTorch. It may limit overall performance so be aware of it.
%env CUBLAS_WORKSPACE_CONFIG=:16:8
```

After that, we can import all the necessary libraries and set up the environment. But first, we will import machine learning libraries that are also necessary.


```python
import os
import time
import pickle
```

Although we already set some env variables, we need to set some additional variables.

In addition to the randomness provided by the random module, Python uses a random seed for its hashing algorithm, which affects how objects are stored in sets and dictionaries. This must happen the same way every time in order for GerryChain runs to be repeatable. The way to accomplish this is to set the [environment variable](https://docs.python.org/3.3/using/cmdline.html) `PYTHONHASHSEED`.

By utilizing cuDNN, Tensorflow/JAX/PyTorch frameworks can take advantage of the specialized hardware and algorithms provided by NVIDIA GPUs to speed up the training and inference of deep learning models. However the downside of this is lack of reproducibility in some cases so we need to not disable it as it is integrated with CUDA. In Tensorflow now you have a function to make everything as determinisitc as possible but in previous version you need to do this: In version 1.14, 1.15 and 2.0 you do it by setting `TF_CUDNN_DETERMINISTIC` to `1`, in 2.1-2.8 version you set `TF_CUDNN_USE_FRONTEND` and `TF_DETERMINISTIC_OPS` to `1`. As I am working now on a newer version of Tensorflow, I am not setting those variables. You can read more about it [here](https://github.com/NVIDIA/framework-determinism/blob/master/doc/tensorflow_status.md).

However, for Jax setting `TF_CUDNN_DETERMINISTIC` to `1` allowed me to get reproducible results, so this variable for Jax should be used:


```python
# All frameworks
os.environ['PYTHONHASHSEED'] = '0'
# Jax
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
```

And finally, it's time for importing our main machine learning modules:


```python
# main imports
import random
import jax
import torch
import numpy as np
import tensorflow as tf

# additional imports
import torchvision
import jax.numpy as jnp
from flax import linen as nn
from flax import serialization as flax_serialization
from flax.training import train_state, checkpoints
import optax
```

In order to ensure that our pipeline is reproducible, the final step is to adjust the parameters in our framework.

One important consideration when setting these parameters is the use of thread parallelism. Thread parallelism refers to the ability of a computer program to execute multiple threads concurrently. In the context of deep learning, this technique can be used to speed up the training process by allowing different threads to work on different parts of the training data at the same time. This can help reduce the overall training time and improve the efficiency of the learning algorithm. However, this also may influence how reproducible are our results.

### Tensorflow

To ensure reproducible results in Tensorflow, it's important to properly configure the threading behavior of the runtime. The `tf.config.threading` module provides APIs for configuring the threading behavior of the TensorFlow runtime. The `tf.config.threading.set_inter_op_parallelism_threads` function specifies the number of threads to use for parallel execution of independent operations (also known as "inter-op" parallelism), while `tf.config.threading.set_intra_op_parallelism_threads` specifies the number of threads to use for parallel execution of operations within a single op (also known as "intra-op" parallelism).

In addition to configuring threading behavior, Tensorflow 2.8 introduced a single function for enabling deterministic operations: `tf.config.experimental.enable_op_determinism()`. According to the Tensorflow documentation, when this function is called, "TensorFlow ops will be deterministic."

For more information on reproducibility in Tensorflow, you can also check out the resources linked in the original [page]((https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism)): the NVIDIA framework determinism [documentation](https://github.com/NVIDIA/framework-determinism/blob/master/doc/tensorflow.md) and the tensorflow-determinism package on [PyPI](https://pypi.org/project/tensorflow-determinism/).

### Pytorch
In the case of thread parallelism, you can avoid setting everything to just one thread, as Pytorch gives a nice solution to this problem, which you will explore later.

There are also a few configs related to cuDNN that you may want to set in order to ensure reproducibility:
- `torch.backends.cudnn.deterministic`: A `bool` that, if set to True, causes cuDNN to only use deterministic convolution algorithms.
- `torch.backends.cudnn.benchmark`: A `bool` that, if set to True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.

In addition to these configs, Pytorch provides a function called `torch.use_deterministic_algorithms`, which "configures PyTorch to use deterministic algorithms instead of nondeterministic ones where available, and to throw an error if an operation is known to be nondeterministic (and without a deterministic alternative)." This can help to ensure reproducibility of your results.

For more information on reproducibility in Pytorch, you can refer to the Pytorch [documentation](https://pytorch.org/docs/stable/notes/randomness.html) on randomness and the NVIDIA framework determinism [documentation](https://github.com/NVIDIA/framework-determinism/blob/master/doc/pytorch.md), which includes a section specifically on Pytorch.

### JAX

Jax is designed to be deterministic, as long as everything is implemented correctly (to my knowledge). In most cases, XLA used by Jax is deterministic on its own. However, there is an issue with cuDNN that can impact the reproducibility of Jax programs. To address this issue, you can set the `TF_CUDNN_DETERMINISTIC` environment variable to `1`, which will make cuDNN deterministic for Jax.

It's worth noting that `TF_CUDNN_DETERMINISTIC` is a Tensorflow environment variable, even though Jax is a separate library. This is because Jax is developed by Google, the same company that developed Tensorflow. As a result, there is some overlap between the two libraries, and you may need to use Tensorflow-specific tools to ensure reproducibility in your Jax programs my padawans 🤓.


```python
################ Tensorflow ################
# set the number of threads running on the CPU
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# set rest of operation to be deterministic
tf.config.experimental.enable_op_determinism()

################ Pytorch ###################
# cudnn settings
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# set rest of operation to be deterministic
torch.use_deterministic_algorithms(mode=True, warn_only=True)
```


```python
print("Devices we will use")
try:
    print(torch.cuda.get_device_name(), torch.cuda.device_count())
    print(tf.config.list_physical_devices("GPU"))
    print(jax.devices())
except:
    print("Well no GPU for you my friend")
```

### What to keep in mind when creating a pipeline

#### Set seed 🌱

Now let's create an array of 2 by 2 random numbers from a Gaussian normal distribution.


```python
# python numbers
python_array = [[random.random() for _ in range(2)] for i in range(2)]
print(f"Python output: {python_array}")
# numpy
numpy_array = np.random.rand(2, 2)
print(f"Numpy output: {numpy_array}")
# pytorch
pytorch_array = torch.rand(2, 2)
print(f"PyTorch output: {pytorch_array}")
# tensorflow
tf_array = tf.random.uniform((2, 2))
print(f"Tensorflow output: {tf_array}")
```

Try to run this code three times, and you will se that each time you will get different results.

So now let's set seed. **Remember** that you need to set seed for all the used packages and it is better to always set seed for `random` and `numpy` packages as you don't know what package generate random numbers in some frameworks. Also you will now see that we will also test JAX as it is the only package where you need to always provide RNGs to every random number creation so you always have reproducible results (if you will not screw up somehow), well this solution have some annoying minuses.

To demonstrate the importance of setting seeds for reproducibility, let's consider a simple example where we generate random numbers using different packages. If you run this code multiple times, you'll notice that the results are different each time. This is because the default behavior of these packages is to generate random numbers based on the current system time, which means that the numbers will be different every time the code is run (I don't know if it is True chatGPT told me that, but results are the same).

To ensure reproducibility, it's important to set seeds for all the packages that generate random numbers. This is especially important for the `random` and `numpy` packages, as you may not always know which package is being used to generate random numbers in a particular framework.

One exception to this rule is JAX, which requires you to provide a random number generator (RNG) for every random number generation. This means that you must always keep track of this variable holding the RNG. While this solution is effective, it can be a little cumbersome as you have to provide an RNG for every random number generation.


```python
# Seed for the random number generator
seed = 0
# seting PRNG for random package
random.seed(seed)
# seting PRNG for numpy
np.random.seed(seed)
# seting PRNG for pytorch
torch.manual_seed(seed)
# if you are on cuda you need to set the seed for the cuda PRNG
torch.cuda.manual_seed(seed)
# seting PRNG for tensorflow
tf.random.set_seed(seed)
# In tensorflow you can set seed for Python, NumPy and TensorFlow with one function
# tf.keras.utils.set_random_seed(seed)
# seting PRNG for jax
rng = jax.random.PRNGKey(seed)
```

And now it's time to test it.


```python
# python numbers
python_array = [[random.random() for _ in range(2)] for i in range(2)]
print(f"Python output: {python_array}")
# numpy
numpy_array = np.random.rand(2, 2)
print(f"Numpy output: {numpy_array}")
# pytorch
pytorch_array = torch.rand(2, 2)
print(f"PyTorch output: {pytorch_array}")
# tensorflow
tf_array = tf.random.uniform((2, 2))
print(f"Tensorflow output: {tf_array}")
# JAX
jax_array = jax.random.uniform(rng, (2, 2))
print(f"JAX output: {jax_array}")
```

If you run the code with those seeds multiple times you'll notice that the results are <span style="color:green">consistent</span> each time. This demonstrates the effectiveness of setting seeds to ensure reproducibility.

It's worth noting that if you re-run the code in a notebook environment, the results may not be consistent because the code is being run in separate cells. To properly test the reproducibility of the code, you should run it multiple times as a single script or restart the notebook each time.

In the case of JAX, it's important to keep track of your RNGs and use them correctly in order to ensure reproducibility. For more information on how to use RNGs in JAX and the advantages of this approach, you can refer to the JAX [documentation](https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html) on random numbers.

### Datasets and Dataloader

As we mentioned earlier, the optimizer updates our deep learning model based on incoming data. If we can make this incoming data deterministic, we can achieve reproducible results from our optimizer.

To ensure reproducibility for datasets and data loaders in Pytorch and Tensorflow, you can use separate generators for each. This will ensure that the same dataset and data loader flow are generated each time the code is run, which can be useful for testing and debugging, as well as for ensuring that the results of a machine learning experiment are reproducible.

It's important to note that JAX does not have a built-in data loader. However, you can still use the techniques described here to ensure reproducibility for your datasets in JAX.

Next, let's take a closer look at the typical usage of datasets and data loaders in Pytorch and Tensorflow.


```python
# Create dummy dataset with pytorch
class DummyPytorchDataset(torch.utils.data.Dataset):
    def __init__(self, shape=(4, 2)):
        self.data = torch.rand(*shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# We create two datasets, one for training and one for testing
pytorch_dataset_train = DummyPytorchDataset()
pytorch_dataset_test = DummyPytorchDataset()

# Create dataloaders
# Train dataloader should be shuffled and test dataloader should not be shuffled
pytorch_dataloader_train = torch.utils.data.DataLoader(
    pytorch_dataset_train, batch_size=2, shuffle=True 
)
pytorch_dataloader_test = torch.utils.data.DataLoader(
    pytorch_dataset_test, batch_size=2, shuffle=False
)
```


```python
# Create dummy dataset with tensorflow
tf_dataset_train = (
    tf.data.Dataset.from_tensor_slices(tf.random.normal((4, 2)))
    .cache()
    .shuffle(4)
    .batch(2)
    .prefetch(tf.data.AUTOTUNE)
)
tf_dataset_test = (
    tf.data.Dataset.from_tensor_slices(tf.random.normal((4, 2)))
    .batch(2)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

# Tensorflow does not have a dataloader, but we can iterate over the dataset or create an iterator
# In this case we will use build in function to iterate over the dataset as numpy arrays
```

Now, have a look at the data for the two epochs.


```python
for epoch in range(2):
    print(f"Epoch {epoch}")
    print("Train dataset with shuffle")
    for i, batch in enumerate(pytorch_dataloader_train):
        print(f"Batch id {i} with data {batch}")

    print("Test dataset without shuffle")
    for i, batch in enumerate(pytorch_dataloader_test):
        print(f"Batch id {i} with data {batch}")
```


```python
for epoch in range(2):
    print(f"Epoch {epoch}")
    print("Train dataset with shuffle")
    for i, batch in enumerate(tf_dataset_train.as_numpy_iterator()):
        print(f"Batch id {i} with data {batch}")

    print("Test dataset without shuffle")
    for i, batch in enumerate(tf_dataset_test.as_numpy_iterator()):
        print(f"Batch id {i} with data {batch}")
```

If you run the code with previous settings and then restart it, you'll get the same results. However, what if you want to go to a specific checkpoint in your dataset and start from there? If you just load the model weights and optimizers and run the code, you'll get the same data as in the beginning of the first epoch, not the saved data from the checkpoint.

In order to reproduce data from a specific checkpoint when training a machine learning model, you can use different approaches depending on the framework you're using. In Tensorflow, you can set the seed for the `shuffle` function for each epoch. In Pytorch, you can use a `torch.Generator` in the `Dataloader`, which handles shuffling and data pre-processing randomness, and can be used to save and load the state of the generator. This allows you to reproduce the data from any point in the training process. To learn more about this technique, you can refer to the Pytorch [documentation](https://pytorch.org/docs/stable/data.html#data-loading-randomness).

Now, let's look at the example:


```python
# Take care that each worker has consistent seed
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Now we create a generator and pass it to the dataloader
g = torch.Generator()
g.manual_seed(seed)

pytorch_dataloader_train = torch.utils.data.DataLoader(
    pytorch_dataset_train, batch_size=2, shuffle=True, worker_init_fn=seed_worker, generator=g,
)
```


```python
# Create dummy dataset with tensorflow without shuffling
tf_dataset_train = (
    tf.data.Dataset.from_tensor_slices(tf.random.normal((4, 2)))
    .cache()
    .batch(2)
    .prefetch(tf.data.AUTOTUNE)
)
```

And now let's see how we can reproduce the same batches of data when shuffling.


```python
pytorch_rng = g.get_state()
for idx, batch in enumerate(pytorch_dataloader_train):
    if idx == 0:
        print(batch)

# now without loading rng state
for idx, batch in enumerate(pytorch_dataloader_train):
    if idx == 0:
        print(batch)

# now we load the rng state
g.set_state(pytorch_rng)
for idx, batch in enumerate(pytorch_dataloader_train):
    if idx == 0:
        print(batch)
```


```python
tf_seed = seed
epoch = 0
for idx, batch in enumerate(tf_dataset_train.shuffle(4, seed=tf_seed).as_numpy_iterator()):
    if idx == 0:
        print(batch)

# now change the seed originally I would do tf_seed+epoch
for idx, batch in enumerate(tf_dataset_train.shuffle(4, seed=tf_seed+1).as_numpy_iterator()):
    if idx == 0:
        print(batch)

# now we load the rng state
for idx, batch in enumerate(tf_dataset_train.shuffle(4, seed=tf_seed).as_numpy_iterator()):
    if idx == 0:
        print(batch)
```

As we mentioned earlier, the state of the random number generator (RNG) is important for reproducing the behavior of a dataset at a specific epoch. Without saving the RNG of the generator, we can't go back to a previous epoch and reproduce the dataset behavior because the shuffling is generated from the RNG, and we need to know its state before shuffling to reproduce it correctly.

By specifying a seed for the shuffling and transformation functions and incrementally changing it, we can keep track of which stage used which seed/RNG. While this is not a perfect solution, it does allow for reproducible results. I will continue to look for a better solution 😉.

To illustrate how to set up reproducible training using different frameworks, let's consider the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset as an example. In the following sections, we'll show you examples of reproducible training in Pytorch, Tensorflow, and JAX, and provide a brief summary of each example, just as a reminder I will not use `hydra` in this notebook but it will be used in pipeline scripts:
<details>
  <summary>PyTorch</summary>

{% highlight python %}
# Ensure once again seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# Gnerator for Dataloaders
g_train = torch.Generator()
g_train.manual_seed(seed)

g_test = torch.Generator()
g_test.manual_seed(seed)


# Parameters for training
n_epochs = 4
num_workers = 4
batch_size_train = 32
batch_size_test = 64
learning_rate = 0.01
momentum = 0.5
log_interval = 1000
cpu = False # NOTE Change if you want to do on cpu/cuda
if cpu:
    device = torch.device("cpu") 
else:
    device = torch.device("cuda") 
{% endhighlight %}


{% highlight python %}
# Create dataloaders
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "/tmp/",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=batch_size_train,
    shuffle=True,
    num_workers=num_workers,
    worker_init_fn=seed_worker, 
    generator=g_train,
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "/tmp/",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    ),
    batch_size=batch_size_test,
    shuffle=False,
    num_workers=num_workers,
    worker_init_fn=seed_worker, 
    generator=g_test,
)
{% endhighlight %}


{% highlight python %}
class Net(torch.nn.Module):
    """Net class for mnist example"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = torch.nn.Dropout2d()
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.relu(torch.nn.functional.max_pool2d(self.conv1(x), 2))
        x = torch.nn.functional.relu(
            torch.nn.functional.max_pool2d(self.conv2_drop(self.conv2(x)), 2)
        )
        x = x.view(-1, 320)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)
{% endhighlight %}


{% highlight python %}
# We create a network and an optimizer
network = Net().to(device)
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
{% endhighlight %}


{% highlight python %}
def train(epoch: int):
    """Train the model

    Args:
        epoch (int): current epoch
    """
    network.train()
    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data.to(device))
        loss = torch.nn.functional.nll_loss(output, target.to(device))
        loss.backward()
        optimizer.step()
    
    end_time = time.time() - start            
    print(
        "Train Epoch: {} {:.4f}s\tLoss: {:.6f}".format(
            epoch,
            end_time,
            loss.item(),
            )
        )


def test(epoch: int):
    """Test the model

    Args:
        epoch (int): current epoch
    """
    network.eval()
    test_loss = 0
    correct = 0
    start = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            target = target.to(device)
            output = network(data.to(device))
            test_loss += torch.nn.functional.nll_loss(
                output, target, reduction='sum'
            ).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().detach().cpu()
    end_time = time.time() - start
    test_loss /= len(test_loader.dataset)
    print(
        "Test set: {:.4f}s Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            end_time,
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    # Save the model
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            'torch_cuda_rng': 0 if cpu else torch.cuda.get_rng_state(),
            "numpy_rng": np.random.get_state(),
            "python_state": random.getstate(),
            "generator_dataloader_train": g_train.get_state(),
            "generator_dataloader_test": g_test.get_state()
        },
        f"/tmp/pytorch_model_{epoch}.pth",
    )
{% endhighlight %}


{% highlight python %}
# We train the model
test(0)
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test(epoch)
{% endhighlight %}

    Test set: 0.7046s Avg. loss: 2.3054, Accuracy: 767/10000 (8%)
    Train Epoch: 1 7.4866s	Loss: 0.306517
    Test set: 0.6440s Avg. loss: 0.1209, Accuracy: 9622/10000 (96%)
    Train Epoch: 2 7.2034s	Loss: 0.109544
    Test set: 0.6896s Avg. loss: 0.0858, Accuracy: 9729/10000 (97%)
    Train Epoch: 3 7.6584s	Loss: 0.171899
    Test set: 0.7642s Avg. loss: 0.0689, Accuracy: 9778/10000 (98%)
    Train Epoch: 4 7.3534s	Loss: 0.313859
    Test set: 0.6776s Avg. loss: 0.0615, Accuracy: 9807/10000 (98%)

{% highlight python %}
# Make sure our model is loaded not from cache
del network, optimizer

# create a new network and optimizer
network = Net().to(device)
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

# Load last checkpoint
checkpoint = torch.load(f"/mnt/data/alzaig/tmp/pytorch_model_{n_epochs}.pth")
epoch_last = checkpoint["epoch"]
network.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
torch.set_rng_state(checkpoint["torch_rng"])
if not cpu:
    torch.cuda.set_rng_state(checkpoint['torch_cuda_rng'])
g_train.set_state(checkpoint["generator_dataloader_train"])
g_test.set_state(checkpoint["generator_dataloader_test"])
np.random.set_state(checkpoint["numpy_rng"])
random.setstate(checkpoint["python_state"])


{% endhighlight %}


{% highlight python %}
test(epoch_last)
for epoch in range(epoch_last + 1, n_epochs + 3):
    train(epoch)
    test(epoch)
{% endhighlight %}

    Test set: 0.6725s Avg. loss: 0.0615, Accuracy: 9807/10000 (98%)
    Train Epoch: 5 7.1801s	Loss: 0.166866
    Test set: 0.7045s Avg. loss: 0.0519, Accuracy: 9839/10000 (98%)
    Train Epoch: 6 7.2071s	Loss: 0.200481
    Test set: 0.6392s Avg. loss: 0.0510, Accuracy: 9835/10000 (98%)


{% highlight python %}
# Make sure our model is loaded not from cache
del network, optimizer

# create a new network and optimizer
network = Net().to(device)
optimizer = torch.optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

# Load second checkpoint
checkpoint = torch.load(f"/mnt/data/alzaig/tmp/pytorch_model_2.pth")
epoch_last = checkpoint["epoch"]
network.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
torch.set_rng_state(checkpoint["torch_rng"])
if not cpu:
    torch.cuda.set_rng_state(checkpoint['torch_cuda_rng'])
g_train.set_state(checkpoint["generator_dataloader_train"])
g_test.set_state(checkpoint["generator_dataloader_test"])
np.random.set_state(checkpoint["numpy_rng"])
random.setstate(checkpoint["python_state"])
{% endhighlight %}


{% highlight python %}
test(epoch_last)
for epoch in range(epoch_last + 1, n_epochs + 1):
    train(epoch)
    test(epoch)
{% endhighlight %}

    Test set: 0.7005s Avg. loss: 0.0858, Accuracy: 9729/10000 (97%)
    Train Epoch: 3 7.3803s	Loss: 0.171899
    Test set: 0.6117s Avg. loss: 0.0689, Accuracy: 9778/10000 (98%)
    Train Epoch: 4 7.8981s	Loss: 0.313859
    Test set: 0.6688s Avg. loss: 0.0615, Accuracy: 9807/10000 (98%)

</details>
Summary time:
- Can we reproduce the results by restarting the script for CPU? Answer ✅.
- Can we reproduce the results by restarting the script for the GPU? Answer ✅
- Can we reproduce the results from the checkpoints? Answer ✅
- Can we reproduce results on CPU and GPU (same device)? Answer ❌
- Can we reproduce results on different CPUs? Answer ❌
- Can we reproduce results on different but same GPU? Answer ✅
- Can we reproduce the results on different graphics cards? Answer ❌ (but I will check this on a few different cards)

<details>
  <summary>Tensorflow</summary>


{% highlight python %}
# Ensure once again seed
tf.keras.utils.set_random_seed(seed)

# Parameters for training
n_epochs = 4
batch_size_train = 16
batch_size_test = 32
learning_rate = 0.01
momentum = 0.5
{% endhighlight %}


{% highlight python %}
# Create a Tensorflow dataset / Keras dataset
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data(path="/tmp/mnist.npz")
x_train, x_test = x_train / 255.0, x_test / 255.0

# Make train and test data loaders
train_loader = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)
test_loader = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .batch(batch_size_test)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)
{% endhighlight %}


{% highlight python %}
# Create Keras model
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Reshape((28,28,-1)),
        tf.keras.layers.Conv2D(filters=10, kernel_size=5),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters=20, kernel_size=5),
        # sorry I did have small error when I used dropout here so I just removed it
        # It will not change enything just for comparison between architectures it will
        tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10),
    ]
)
# Compile model
model.compile()
{% endhighlight %}


{% highlight python %}
# Define loss and optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)

# Create checkpoint callback
cp = tf.train.Checkpoint(model=model, optimizer=optimizer)

# Metrics
train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
test_loss = tf.keras.metrics.Mean(name="test_loss")
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")
{% endhighlight %}


{% highlight python %}
@tf.function
def train_step(data, target):
    """Train function
    
    Args:
        data: input data
        target: target for the input data
    """
    with tf.GradientTape() as tape:
        predictions = model(data, training=True)
        loss = loss_fn(target, predictions)
    grad = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grad, model.trainable_variables))

    train_loss(loss)
    train_accuracy(target, predictions)
    return

@tf.function
def test_step(data, target):
    """Train function
    
    Args:
        data: input data
        target: target for the input data
    """
    predictions = model(data, training=False)
    t_loss = loss_fn(target, predictions)
    test_loss(t_loss)
    test_accuracy(target, predictions)
    return
{% endhighlight %}


{% highlight python %}
# Evaluate
test_loss.reset_states()
test_accuracy.reset_states()
for images, labels in test_loader:
    test_step(images, labels)
print(
    f"Epoch {0}, "
    f"Test Loss: {test_loss.result():.4f}, "
    f"Test Accuracy: {test_accuracy.result():.4f}"
)

for epoch in range(1, n_epochs + 1):
    # Reset trackers for metrics
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    dl_seed = seed + epoch

    # Train
    for batch, (images, labels) in enumerate(train_loader.shuffle(len(x_train), seed=dl_seed).batch(batch_size_train)):
        train_step(images, labels)
    
    # Save
    cp.write(f"/tmp/tf_models_{epoch}.h5")
    
    # Test
    for images, labels in test_loader:
        test_step(images, labels)

    print(
        f"Epoch {epoch}, "
        f"Loss: {train_loss.result():.4f}, "
        f"Accuracy: {train_accuracy.result():.4f}, "
        f"Test Loss: {test_loss.result():.4f}, "
        f"Test Accuracy: {test_accuracy.result():.4f}"
    )
{% endhighlight %}

    Epoch 0, Test Loss: 2.3172, Test Accuracy: 0.0897
    Epoch 1, Loss: 0.2931, Accuracy: 0.9081, Test Loss: 0.0746, Test Accuracy: 0.9767
    Epoch 2, Loss: 0.1088, Accuracy: 0.9676, Test Loss: 0.0658, Test Accuracy: 0.9786
    Epoch 3, Loss: 0.0832, Accuracy: 0.9746, Test Loss: 0.0440, Test Accuracy: 0.9855
    Epoch 4, Loss: 0.0687, Accuracy: 0.9787, Test Loss: 0.0434, Test Accuracy: 0.9860

{% highlight python %}
# Load last epoch
cp.restore(f"/tmp/tf_models_{n_epochs}.h5")

# Evaluate
test_loss.reset_states()
test_accuracy.reset_states()
for images, labels in test_loader:
    test_step(images, labels)
print(
    f"Epoch {n_epochs}, "
    f"Test Loss: {test_loss.result():.4f}, "
    f"Test Accuracy: {test_accuracy.result():.4f}"
)

for epoch in range(n_epochs + 1, n_epochs + 3):
    # Reset trackers for metrics
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    dl_seed = seed + epoch

    # Train
    for batch, (images, labels) in enumerate(train_loader.shuffle(len(x_train), seed=dl_seed).batch(batch_size_train)):
        train_step(images, labels)
    
    # Save
    cp.write(f"/tmp/tf_models_{epoch}.h5")
    
    # Test
    for images, labels in test_loader:
        test_step(images, labels)

    print(
        f"Epoch {epoch}, "
        f"Loss: {train_loss.result():.4f}, "
        f"Accuracy: {train_accuracy.result():.4f}, "
        f"Test Loss: {test_loss.result():.4f}, "
        f"Test Accuracy: {test_accuracy.result():.4f}"
    )
{% endhighlight %}

    Epoch 4, Test Loss: 0.0434, Test Accuracy: 0.9860
    Epoch 5, Loss: 0.0587, Accuracy: 0.9822, Test Loss: 0.0362, Test Accuracy: 0.9883
    Epoch 6, Loss: 0.0510, Accuracy: 0.9844, Test Loss: 0.0371, Test Accuracy: 0.9866

{% highlight python %}
# Load second epoch
cp.restore("/tmp/tf_models_2.h5")

# Evaluate
test_loss.reset_states()
test_accuracy.reset_states()
for images, labels in test_loader:
    test_step(images, labels)
print(
    f"Epoch {2}, "
    f"Test Loss: {test_loss.result():.4f}, "
    f"Test Accuracy: {test_accuracy.result():.4f}"
)

for epoch in range(2 + 1, n_epochs + 1):
    # Reset trackers for metrics
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    dl_seed = seed + epoch

    # Train
    for batch, (images, labels) in enumerate(train_loader.shuffle(len(x_train), seed=dl_seed).batch(batch_size_train)):
        train_step(images, labels)
    
    # Save
    cp.write(f"/tmp/tf_models_{epoch}.h5")
    
    # Test
    for images, labels in test_loader:
        test_step(images, labels)

    print(
        f"Epoch {epoch}, "
        f"Loss: {train_loss.result():.4f}, "
        f"Accuracy: {train_accuracy.result():.4f}, "
        f"Test Loss: {test_loss.result():.4f}, "
        f"Test Accuracy: {test_accuracy.result():.4f}"
    )
{% endhighlight %}

    Epoch 2, Test Loss: 0.0658, Test Accuracy: 0.9786
    Epoch 3, Loss: 0.0841, Accuracy: 0.9743, Test Loss: 0.0433, Test Accuracy: 0.9862
    Epoch 4, Loss: 0.0683, Accuracy: 0.9792, Test Loss: 0.0397, Test Accuracy: 0.9874

</details>
Summary time:
- Can we reproduce the results by restarting the script for CPU? Answer ✅.
- Can we reproduce the results by restarting the script for the GPU? Answer ✅
- Can we reproduce the results from the checkpoints? Answer ❌
- Can we reproduce results on CPU and GPU (same device)? Answer ❌
- Can we reproduce results on different CPUs? Answer ❌
- Can we reproduce results on different but same GPU? Answer ✅
- Can we reproduce the results on different graphics cards? Answer ❌ (but I will check it on a few different cards)

<details>
  <summary>JAX</summary>

For our example of reproducible training in JAX, we'll be using the [Flax](https://flax.readthedocs.io/en/latest/) library to build our deep neural network (DNN). Flax is an overlay library built on top of JAX that provides a more intuitive interface for building DNNs. It does not add any additional functionality to JAX, but it makes it easier to build DNNs using JAX, which was not designed specifically for this purpose.

{% highlight python %}
# Ensure once again seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
rng = jax.random.PRNGKey(seed)

# Gnerator for Dataloaders
g_train = torch.Generator()
g_train.manual_seed(seed)

g_test = torch.Generator()
g_test.manual_seed(seed)

# Parameters for training
n_epochs = 4
num_workers = 4
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
{% endhighlight %}


{% highlight python %}
class CNN(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x, training=False):
    x = nn.Conv(features=10, kernel_size=(5, 5))(x)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.relu(x)
    x = nn.Conv(features=20, kernel_size=(5, 5))(x)
    x = nn.Dropout(0.2)(x, deterministic=not training)
    x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.relu(x)
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=50)(x)
    x = nn.relu(x)
    x = nn.Dropout(0.2)(x, deterministic=not training)
    x = nn.Dense(features=10)(x)
    return x
{% endhighlight %}


{% highlight python %}
def cross_entropy_loss(logits, labels):
    """Compute the cross-entropy loss given logits and labels.

    Args:
        logits: output of the model
        labels: labels of the data
    """
    labels_onehot = jax.nn.one_hot(labels, num_classes=10)
    return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()


def compute_metrics_jax(logits, labels):
    """Compute metrics for the model.

    Args:
        logits: output of the model
        labels: labels of the data
    """
    loss = cross_entropy_loss(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return {'loss': loss, 'accuracy': accuracy}


def create_train_state(rng, learning_rate, momentum):
    """Creates initial `TrainState` with `sgd`."""
    cnn = CNN()
    rng, rng_dropout = jax.random.split(rng)
    params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params'] # initialize parameters by passing a template image
    tx = optax.sgd(learning_rate, momentum)
    return train_state.TrainState.create(
        apply_fn=cnn.apply, params=params, tx=tx)
{% endhighlight %}


{% highlight python %}
@jax.jit
def train_step(state, image, label, dropout_rng):
  """Train for a single step. Also jit-compiled for speed.
  
  Args:
    params: parameters of the model
    image: input image
    label: label of the image
    dropout_rng: rng for dropout
  """
  def loss_fn(params):
    logits = CNN().apply({'params': params}, x=image, training=True, rngs={'dropout': dropout_rng})
    loss = cross_entropy_loss(logits=logits, labels=label)
    return loss, logits
  grad_fn = jax.grad(loss_fn, has_aux=True)
  grads, logits = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = compute_metrics_jax(logits=logits, labels=label)
  return state, metrics


@jax.jit
def eval_step(params, image, label):
  """Evaluate for a single step. Also jit-compiled for speed.
  
  Args:
    params: parameters of the model
    image: input image
    label: label of the image
  """
  logits = CNN().apply({'params': params}, image)
  return compute_metrics_jax(logits=logits, labels=label)
{% endhighlight %}


{% highlight python %}
def train_epoch(state, train_ds, epoch, rng):
  """Train for a single epoch.
  
  Args:
    params: model parameters
    test_ds: test dataloader
    epoch: current epoch
  """
  batch_metrics = []
  start = time.time()
  for batch_idx, (data, target) in enumerate(train_ds):
    rng, rng_dropout = jax.random.split(rng)
    state, metrics = train_step(state, data, target, rng_dropout)
    batch_metrics.append(metrics)

  # compute mean of metrics across each batch in epoch.
  batch_metrics_np = jax.device_get(batch_metrics)
  epoch_metrics_np = {
      k: np.mean([metrics[k] for metrics in batch_metrics_np])
      for k in batch_metrics_np[0]} # jnp.mean does not work on lists

  end_time = time.time() - start
  print('train epoch: %d, time %.4f,, loss: %.4f, accuracy: %.2f' % (
      epoch, end_time, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100))
  return state, rng


def eval_model(params, test_ds):
  """Evaluate the model.
  
  Args:
    params: model parameters
    test_ds: test dataloader
  """
  batch_metrics = []
  start = time.time()
  for batch_idx, (data, target) in enumerate(test_ds):
    metrics = eval_step(params, data, target)
    batch_metrics.append(metrics)
    
  # compute mean of metrics across each batch in epoch.
  batch_metrics_np = jax.device_get(batch_metrics)
  metrics = {
      k: np.mean([metrics[k] for metrics in batch_metrics_np])
      for k in batch_metrics_np[0]} # jnp.mean does not work on lists
  end_time = time.time() - start
  summary = jax.tree_util.tree_map(lambda x: x.item(), metrics) # map the function over all leaves in metrics
  return summary['loss'], summary['accuracy'], end_time
{% endhighlight %}


{% highlight python %}
# Transformations applied on each image => bring them into a numpy array
def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    img = np.transpose(img, (1, 2, 0))
    return img

# We need to stack the batch elements
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "/mnt/data/alzaig/tmp/",
        train=True,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                image_to_numpy
            ]
        ),
    ),
    batch_size=batch_size_train,
    shuffle=True,
    collate_fn=numpy_collate,
    num_workers=num_workers,
    worker_init_fn=seed_worker, 
    generator=g_train,
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        "/mnt/data/alzaig/tmp/",
        train=False,
        download=True,
        transform=torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                image_to_numpy
            ]
        ),
    ),
    batch_size=batch_size_test,
    shuffle=False,
    collate_fn=numpy_collate,
    num_workers=num_workers,
    worker_init_fn=seed_worker, 
    generator=g_test,
)
{% endhighlight %}


{% highlight python %}
rng, init_rng = jax.random.split(rng)
state = create_train_state(init_rng, learning_rate, momentum)
{% endhighlight %}


{% highlight python %}
test_loss, test_accuracy, end_time = eval_model(state.params, test_loader)
print(' test epoch: %d, time: %.4f, loss: %.2f, accuracy: %.2f' % (
      0, end_time, test_loss, test_accuracy * 100))

for epoch in range(1, n_epochs + 1):
  state, rng = train_epoch(state, train_loader, epoch, rng)
  # Evaluate on the test set after each training epoch
  test_loss, test_accuracy, end_time = eval_model(state.params, test_loader)
  print(' test epoch: %d, time: %.4f, loss: %.2f, accuracy: %.2f' % (
      epoch, end_time, test_loss, test_accuracy * 100))
  # Save the model parameters
  numpy_rng = np.random.get_state()
  python_state = random.getstate()
  torch_state = torch.get_rng_state() # well only this is required but for the sake of completeness I am adding all
  checkpoints.save_checkpoint(
      ckpt_dir="/mnt/data/alzaig/tmp/jax_checkpoints_{}".format(epoch),
      target={"state": state,
              "epoch": epoch,
              "rng": rng,
              "pytorch_rng": torch_state.numpy(),
              "numpy_rng": numpy_rng,
              "python_state": python_state,
              "generator_dataloader_train": g_train.get_state().numpy(),
              "generator_dataloader_test": g_test.get_state().numpy(),
            },
            step=epoch,
            overwrite=True,
    )
{% endhighlight %}

    test epoch: 0, time: 1.6248, loss: 2.51, accuracy: 7.01
    train epoch: 1, time 5.7122,, loss: 0.3448, accuracy: 89.40
    test epoch: 1, time: 0.8474, loss: 0.10, accuracy: 96.94
    train epoch: 2, time 2.9587,, loss: 0.1360, accuracy: 95.87
    test epoch: 2, time: 0.8584, loss: 0.07, accuracy: 97.87
    train epoch: 3, time 2.8872,, loss: 0.1028, accuracy: 96.83
    test epoch: 3, time: 0.8345, loss: 0.05, accuracy: 98.38
    train epoch: 4, time 3.0020,, loss: 0.0866, accuracy: 97.41
    test epoch: 4, time: 0.8817, loss: 0.05, accuracy: 98.62

{% highlight python %}
# Load the model parameters
load_dict = checkpoints.restore_checkpoint(ckpt_dir=f"/mnt/data/alzaig/tmp/jax_checkpoints_{n_epochs}", target=None)
rng = load_dict["rng"]
torch.set_rng_state(torch.tensor(load_dict["pytorch_rng"]))
python_state = list(load_dict["python_state"].values())
python_state[1] = tuple(python_state[1].values())
g_train.set_state(torch.tensor(load_dict["generator_dataloader_train"]))
g_test.set_state(torch.tensor(load_dict["generator_dataloader_test"]))
np.random.set_state(tuple(load_dict["numpy_rng"].values()))
random.setstate(python_state)
epoch_last = load_dict["epoch"]
state = flax_serialization.from_state_dict(state, load_dict['state'])
{% endhighlight %}


{% highlight python %}
test_loss, test_accuracy, end_time = eval_model(state.params, test_loader)
print(' test epoch: %d, time: %.4f, loss: %.2f, accuracy: %.2f' % (
      epoch_last, end_time, test_loss, test_accuracy * 100))

for epoch in range(epoch_last+1, n_epochs + 3):
  state, rng = train_epoch(state, train_loader, epoch, rng)
  # Evaluate on the test set after each training epoch
  test_loss, test_accuracy, end_time = eval_model(state.params, test_loader)
  print(' test epoch: %d, time: %.4f, loss: %.2f, accuracy: %.2f' % (
      epoch, end_time, test_loss, test_accuracy * 100))
  # Save the model parameters
  numpy_rng = np.random.get_state()
  python_state = random.getstate()
  torch_state = torch.get_rng_state() # well only this is required but for the sake of completeness I am adding all
  checkpoints.save_checkpoint(
      ckpt_dir="/mnt/data/alzaig/tmp/jax_checkpoints_{}".format(epoch),
      target={"state": state,
              "epoch": epoch,
              "rng": rng,
              "pytorch_rng": torch_state.numpy(),
              "numpy_rng": numpy_rng,
              "python_state": python_state,
              "generator_dataloader_train": g_train.get_state().numpy(),
              "generator_dataloader_test": g_test.get_state().numpy(),
            },
            step=epoch,
            overwrite=True,
    )
{% endhighlight %}

    test epoch: 4, time: 0.8676, loss: 0.05, accuracy: 98.62
    train epoch: 5, time 4.7443,, loss: 0.0759, accuracy: 97.73
    test epoch: 5, time: 0.8361, loss: 0.04, accuracy: 98.72
    train epoch: 6, time 3.1997,, loss: 0.0660, accuracy: 97.96
    test epoch: 6, time: 0.8729, loss: 0.04, accuracy: 98.92

{% highlight python %}
# Load the second model parameters
load_dict = checkpoints.restore_checkpoint(ckpt_dir=f"/mnt/data/alzaig/tmp/jax_checkpoints_2", target=None)
rng = load_dict["rng"]
torch.set_rng_state(torch.tensor(load_dict["pytorch_rng"]))
python_state = list(load_dict["python_state"].values())
python_state[1] = tuple(python_state[1].values())
g_train.set_state(torch.tensor(load_dict["generator_dataloader_train"]))
g_test.set_state(torch.tensor(load_dict["generator_dataloader_test"]))
np.random.set_state(tuple(load_dict["numpy_rng"].values()))
random.setstate(python_state)
epoch_last = load_dict["epoch"]
state = flax_serialization.from_state_dict(state, load_dict['state'])
{% endhighlight %}


{% highlight python %}
test_loss, test_accuracy, end_time = eval_model(state.params, test_loader)
print(' test epoch: %d, time: %.4f, loss: %.2f, accuracy: %.2f' % (
      epoch_last, end_time, test_loss, test_accuracy * 100))

for epoch in range(epoch_last+1, n_epochs + 1):
  state, rng = train_epoch(state, train_loader, epoch, rng)
  # Evaluate on the test set after each training epoch
  test_loss, test_accuracy, end_time = eval_model(state.params, test_loader)
  print(' test epoch: %d, time: %.4f, loss: %.2f, accuracy: %.2f' % (
      epoch, end_time, test_loss, test_accuracy * 100))
  # Save the model parameters
  numpy_rng = np.random.get_state()
  python_state = random.getstate()
  torch_state = torch.get_rng_state() # well only this is required but for the sake of completeness I am adding all
  checkpoints.save_checkpoint(
      ckpt_dir="/mnt/data/alzaig/tmp/jax_checkpoints_{}".format(epoch),
      target={"state": state,
              "epoch": epoch,
              "rng": rng,
              "pytorch_rng": torch_state.numpy(),
              "numpy_rng": numpy_rng,
              "python_state": python_state,
              "generator_dataloader_train": g_train.get_state().numpy(),
              "generator_dataloader_test": g_test.get_state().numpy(),
            },
            step=epoch,
            overwrite=True,
    )
{% endhighlight %}

    test epoch: 2, time: 0.8855, loss: 0.07, accuracy: 97.87
    train epoch: 3, time 2.9279,, loss: 0.1028, accuracy: 96.83
    test epoch: 3, time: 0.8658, loss: 0.05, accuracy: 98.38
    train epoch: 4, time 2.9546,, loss: 0.0866, accuracy: 97.41
    test epoch: 4, time: 0.8714, loss: 0.05, accuracy: 98.62

</details>
Summary time:
- Can we reproduce the results by restarting the script for CPU? Answer ✅.
- Can we reproduce the results by restarting the script for the GPU? Answer ✅
- Can we reproduce the results from the checkpoints? Answer ✅
- Can we reproduce results on CPU and GPU (same device)? Answer ❌
- Can we reproduce results on different CPUs? Answer ❌
- Can we reproduce results on different but same GPU? Answer ✅
- Can we reproduce the results on different graphics cards? Answer ❌ (but I will check it on a few different cards)

### (Optional) Time to clean up before the summary


```python
# You may want to check if there is nothing valuable in the directory before deleting it
# I even recommend you to do that checking everything in the directory before deleting it
%rm -rf /tmp/*
```

## Summary

In this blog, we explored the issue of reproducibility in the popular machine learning frameworks Jax, Tensorflow, and Pytorch. We discussed the challenges and solutions for achieving reproducibility in these frameworks, as well as the benefits of using reproducible methods in machine learning research. We also noted that this blog post did not cover reproducibility in distributed learning environments, which will be the focus of a separate blog post. 

One of the key takeaways from this blog is the importance of setting seeds in order to achieve reproducible results. This is a minimal requirement in the industry, but there are other approaches that can also be useful, particularly in the context of research comparisons. However, it's important to keep in mind that some of these approaches may come at the cost of reduced performance, so they may not be ideal in all situations. That being said, reproducibility can be particularly useful for debugging purposes, even if it may not be the most efficient option in terms of performance.

In this blog post, we summarized the current state of reproducibility in these frameworks, including the ability to reproduce results by restarting the script on both CPU and GPU, using checkpoints, and on different hardware configurations. We also emphasized the importance of understanding the hardware and software environment in which the code was run, as this can significantly impact reproducibility. Lastly, we demonstrated how to create a reproducible MLCube pipeline for these codes.

## Pipeline

Right now you should have this in your pipeline directory:
```text
reproduc-ml-tutorial/
  Dockerfile
  requirements.txt
  Singularity.recipe
  mlcube.yaml
```
In this blog post, we will be adding four scripts (three python scripts and one Bash script) to our project. These scripts will be used to train a machine learning model on the MNIST dataset using three different frameworks: Tensorflow, PyTorch, and JAX:
- mnist_tensorflow.py - where we implement Tensorflow version of MNIST training
- mnist_pytorch.py - where we implement PyTorch version of MNIST training
- mnist_jax.py - where we implement JAX version of MNIST training

We will also add a `workspace` directory to our project. This directory will be used to store input/output file system artifacts such as data sets, configuration files, and log files. Having all of these files in one place makes it easier to run our code on remote hosts and then sync the results back to our local machines.

To help us manage and track the various parameters used in our code, we will be using the `hydra` library. The parameters for our scripts will be defined in YAML files located in the workspace directory.

Finally, the Bash script will be used to control which script gets run based on the task at hand. We will not be discussing the implementation of these scripts in this blog post, but you can check out the accompanying GitHub [repository](https://github.com/WolodjaZ/reproduc-ml-tutorial) to see how we implemented them and used with this Bash script:
```bash
#!/bin/bash

# Set workspace env to use in configs
workspace_path=${2##*=}
export WORKSPACE_PATH=$workspace_path

# Case statement which file with dedicated framework to run
case "$1" in
  pytorch)
    python /workspace/mnist_pytorch.py ;;
  tensorflow)
    python /workspace/mnist_tensorflow.py ;;
  jax)
    python /workspace/mnist_jax.py ;;
  *)
    echo "Sorry, you need to give one of this string `pytorch`, `tensorflow`, `jax` not $1" ;;
esac
```
The final addition to our project will be a file called `train.yaml` that is located in the `workspace` directory. This file will contain the parameters for training each of our scripts. It will be imported whenever we build Singularity or Docker images for our project.
```YAML
# Hydra configs
hydra:
  output_subdir: null # Disable Hydra subdir creation 
  run:
    dir: ${oc.env:WORKSPACE_PATH}/outputs/${hydra.job.name}/${now:%Y-%m-%d_%H-%M-%S} # Path where Hydra should store logs 
    
# Parameters for the pipeline
params:
  seed: 0 # Seed value
  n_epochs: 4 # Number of epoch
  num_workers: 4 # Number of workers for Dataloaders
  batch_size_train: 32 # Batch size for train dataset
  batch_size_test: 64 # Batch size for test dataset
  optimizer: sgd # Optimizer name
  learning_rate: 0.01 # Learning rate for optimizer
  momentum: 0.5 # Momentum also parameter for optimizer
  log_interval: 1000 # How often should logs be made in training loop

# data path
output:
  data_path: ${oc.env:WORKSPACE_PATH}/data # Path where data should be stored
  model_path: ${oc.env:WORKSPACE_PATH}/model # Path where models should be created
```
And `.dockerignore` to ignore all the data from `workspace` that will be additionally created:
```text
# Ignore everything in workspace directory
workspace/
# Ignore virtual environment
venv/

# except these two configuration files
!/workspace/train.yaml
```
Now we have our directory looking something like that:
```text
reproduc-ml-tutorial/
  workspace/          # Default location for data sets, logs, models, parameter files.
    train.yaml        #   Train hyper-parameters.
  .dockerignore       # Docker ignore file that prevents workspace directory to be sent to docker server.
  DockerBuildfile     # Docker recipe.
  index.ipynb         # Example notebook from Reproducibility in Tensorflow/PyTorch/JAX part 2
  mlcube.yaml         # MLCube definition file.
  train_jax.py        # Python source code training simple neural network using MNIST data set with JAX.
  train_pytorch.py    # Python source code training simple neural network using MNIST data set with PyTorch.
  train_tensorflow.py # Python source code training simple neural network using MNIST data set with Tensorflow.
  requirements.txt    # Python project dependencies.
  run.sh              # Main bash script that lunches python script based on passed argument
  Singularity.recipe  # Singularity recipe.
```

## Running MLCube

In this section, we will go over how to run MLCube using either the `Docker` runner or the `Singularity` runner.

### Docker Runner

To configure MLCube for use with the Docker runner (this step is optional, as the Docker runner will automatically run the configure phase if the image does not already exist), run the following command:
```bash
mlcube configure --mlcube=. --platform=docker
```
To run the three tasks (PyTorch model training, Tensorflow model training, and JAX model training), use the following commands:
```bash
mlcube run --mlcube=. --platform=docker --task=pytorch
mlcube run --mlcube=. --platform=docker --task=tensorflow
mlcube run --mlcube=. --platform=docker --task=jax
```

### Singularity Runner

To configure MLCube for use with the Singularity runner, run the following command:
```bash
mlcube configure --mlcube=. --platform=singularity
```
To run the three tasks (PyTorch model training, Tensorflow model training, and JAX model training), use the following commands:
```bash
mlcube run --mlcube=. --platform=singularity --task=pytorch
mlcube run --mlcube=. --platform=singularity --task=tensorflow
mlcube run --mlcube=. --platform=singularity --task=jax
```

## Wrap up 🏁

If you have followed along with the first part of this tutorial and completed all of the steps in the second part, you should now have a good understanding of how to create reproducible AI projects. It is important to remember everything you have learned, as it will be useful in the future. The repository for the code in this tutorial can be found [here](https://github.com/WolodjaZ/reproduc-ml-tutorial).

## Acknowledgment

I would like to express my gratitude to all the writers and developers who have worked on implementing and documenting reproducible solutions for the various frameworks. In particular, I would like to thank the developers at [OpenAI]((https://openai.com)) for creating ChatGPT, which has made writing this blog much easier and MLCommons community for working hard to help everyone make better ML solution.

![chatGPT](/images/posts/chatGPT_meme.png)

## References:

- Python documentation on using the command line (PYTHONHASHSEED): https://docs.python.org/3.3/using/cmdline.html
- MLCube documentation: https://mlcommons.github.io/mlcube/
- For more guidance on reproducible research, check out The Turing Way: https://the-turing-way.netlify.app/reproducible-research/reproducible-research.html
### Pytorch
- Pytorch documentation on randomness: https://pytorch.org/docs/stable/notes/randomness.html
- Pytorch determinism documentation: https://github.com/NVIDIA/framework-determinism/blob/master/doc/pytorch.md
### Tensorflow
- Tensorflow issue on reproducibility: https://github.com/tensorflow/tensorflow/issues/53771
- Tensorflow documentation on op determinism: https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism
- Tensorflow determinism tutorial: https://suneeta-mall.github.io/2019/12/22/Reproducible-ml-tensorflow.html
- Tensorflow determinism package: https://pypi.org/project/tensorflow-determinism/
- Tensorflow determinism documentation: https://github.com/NVIDIA/framework-determinism/blob/master/doc/tensorflow.md
- Tensorflow determinism status documentation: https://github.com/NVIDIA/framework-determinism/blob/master/doc/tensorflow_status.md
- Tensorflow documentation on threading: https://www.tensorflow.org/api_docs/python/tf/config/threading
### Jax
- Jax documentation on random numbers: https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html
- Flax issue on reproducibility: https://github.com/google/flax/issues/33


