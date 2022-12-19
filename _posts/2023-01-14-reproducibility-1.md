---
title: "Reproducibility in Tensorflow/PyTorch/JAX Part 1/2"
excerpt: "In this blog, we discuss the importance of reproducibility in machine learning and provide tips on how to ensure reproducibility of your models in TensorFlow, PyTorch, and JAX. Part 1/2 of the blog about reproducibility."
data: 2023-01-14
languages: [python, jax, tensorflow, pytorch]
tags: [Reproducibility, JAX, Tensorflow, PyTorch]
---

If you are reading this, I am probably dead 💀 ...

<img src="../images/posts/reproducibility_meme.JPG" width="350">

Please keep in mind that this post is not finished as there are still unresolved topics about reproducibility that I will try to trace 🕵. If you see any mistakes, please email me with an explanation, and I will try to fix it. So, after this brief introduction, shall we begin?

Before diving into my blog, I recommend reading about reproducibility in research in [The Turing Way](https://the-turing-way.netlify.app/reproducible-research/reproducible-research.html). The Turing Way is an open-source, collaborative, and community-driven "book" that aims to provide all the information researchers and data scientists in academia, industry, and the public sector need to ensure their projects are easy to reproduce and reuse. While there is a lot of information to read, it is not necessary to read everything. I suggest focusing on the section about reproducibility. Note that The Turing Way does not cover reproducibility in Tensorflow/PyTorch/JAX, but that is what my blog is for.

Reproducibility is a vital element of scientific research, as it enables others to confirm and build upon previous findings. This is particularly relevant in the realm of machine learning, where reproducibility allows for more accurate comparisons between different methods and algorithms. In this blog post, we will examine the issue of reproducibility in the popular machine learning frameworks [Jax](https://jax.readthedocs.io/en/latest/), [Tensorflow](https://www.tensorflow.org), and [Pytorch](https://pytorch.org). We will explore the challenges and solutions for achieving reproducibility in these frameworks as well as the benefits of using reproducible methods in machine learning research.

This blog is divided into two parts. In the first part, we will discuss theoretical topics related to reproducibility, including the why, what, and how. In the second part, we will provide practical examples in each of the frameworks, demonstrating how to reproduce the concepts discussed in the first part and achieve reproducibility. Additionally, we will utilize [MLCube](https://mlcommons.org/en/mlcube/) from [MLCommons](https://mlcommons.org/en/) to further enhance reproducibility in our examples. It's worth noting that this blog post will not cover reproducibility in distributed learning environments. We will address this topic in a separate blog post 📜 focused specifically on distributed learning and reproducibility.

In the industry, machine learning engineers rely on reproducible results every time they run code. This enables them to improve their models, explore different hyperparameter settings, and facilitate debugging. In the research field, reproducibility is particularly important for accurately comparing results to those published in papers. In this blog post, we will delve into the importance of reproducibility in both the industry and research environment. **It's important to note that will be trying to reproduce results, not time performance, as it is more possible to do**.


## MLCube

MLCube is a set of best practices for creating ML software that can be easily used on a variety of systems. It makes it easier for researchers to share innovative ML models, for developers to experiment with different models, and for software companies to create infrastructure for models. MLCube aims to put ML in the hands of more people and create opportunities for broader adoption.

MLCube is not a new framework or service, but rather a consistent interface for machine learning models in containers like Docker. MLCommons provides open source "runners" that enable MLCube models to be trained with a single command on local machines, major clouds, or Kubernetes clusters. Additionally, MLCube is designed to make it easy to build new infrastructure based on the interface.

The mission of MLCommons is to improve machine learning for everyone. MLCommons, along with its 50+ founding Members and Affiliates, is working to advance machine learning from a research field into a mature industry through the use of benchmarks, public datasets, and best practices. These efforts aim to accelerate machine learning innovation and bring the benefits of ML to a wide range of industries and applications, such as healthcare, automotive safety, and natural language processing.

I hope in the future, when you read this blog, I will be a member of this group of dedicated and devoted people, but I also encourage you to look at this community and what they do, and even try to participate 🫵. The more people, the more fun there always is 😎 .

## Randomness in deep learning

In deep learning, randomness can arise from several sources, such as the **initialization of weights** in a neural network or the use of **random samples** from the **training dataset** in the stochastic gradient descent (SGD) algorithm. To achieve reproducibility in deep learning, it is important to consider these sources of randomness and how they can be controlled. This includes processes such as:
- weight initialization 
- dataset and data loaders
- stochastic operations like dropout (on paper they are stochastic).

However, reproducibility can also be affected by the environment in which the model is run, including the versions of packages and libraries, the version of Python, and the hardware used. For example, using different versions of packages or libraries can alter the behavior of certain functions and impact the reproducibility of results. Similarly, using different hardware or optimization tools like [CUDA](https://developer.nvidia.com/cuda-downloads) can also affect reproducibility.

To overcome these challenges, it is important to carefully consider and document the environment in which the model was trained and ensure that it is replicated when the model is run by others.

### Random number generators (PRNG)

Random number generators (RNGs) are an important component of machine learning frameworks like Pytorch, Tensorflow, and Jax, as well as scientific computing libraries like [Numpy](https://numpy.org). These RNGs use pseudorandom number generators (PRNGs) to produce sequences of numbers that are statistically close to random.

PRNGs are based on algorithms that generate a sequence of numbers using a fixed starting point, called the seed 🌱. The generated numbers are determined by the seed and the algorithm used, so they will be the same every time the code is run with the same seed. This can be useful for reproducibility, as it allows you to obtain the same results by setting the same seed. However, it also means that the generated numbers are not truly random and may not be suitable for certain applications. I advise to look into this lecture from [the university of Utah](https://www.math.utah.edu/~alfeld/Random/Random.html) and this [paper](http://www.thesalmons.org/john/random123/papers/random123sc11.pdf).

There are various algorithms used by PRNGs, including Monte Carlo techniques and counter-based RNGs. The Python `random` module uses the [Mersenne Twister algorithm](https://www.sciencedirect.com/topics/computer-science/mersenne-twister), which uses a state of `624` integers to generate a sequence of seemingly random numbers. Numpy's `numpy.random` and Pytorch's `torch.rand` packages also use the Mersenne Twister algorithm. Tensorflow's `tensorflow.random` package uses the [Philox algorithm](https://numpy.org/doc/stable/reference/random/bit_generators/philox.html), which is based on counter-based RNGs. Jax's `jax.random` package uses the [ThreeFry](https://bashtage.github.io/randomgen/bit_generators/threefry.html), which is also based on counter-based RNGs algorithm.

It is important to keep in mind that these RNGs are not truly random, and they may not be suitable for all applications. If you need truly random numbers, you may need to use a hardware-based RNG, such as a physical device that generates random numbers based on quantum phenomena.

### Environment reproducibility

To ensure reproducibility in your deep learning projects, it is important to carefully manage your environment. Here are some tips for making your environment reproducible:
1. Use a tool like [pyenv](https://realpython.com/intro-to-pyenv/) to manage your Python version. This will ensure that you are using the same version of Python across different environments.
2. Create virtual environments for your projects using tools like [venv](https://docs.python.org/3/library/venv.html), [Poetry](https://python-poetry.org), [Conda](https://conda-forge.org), or [pdm](https://github.com/pdm-project/pdm). This will allow you to isolate your project dependencies and avoid conflicts with other packages. To create `venv` just run in shell `python -m venv venv` and voilà. To activate it run `source venv/bin/activate`.
3. Keep track of the packages and their versions that you use in your virtual environments. You can use the `pip freeze` command to generate a `requirements.txt` file that lists all of the packages and their versions `pip freeze > requirements.txt`. This will help you recreate the exact same environment later when running `pip install -r requirements.txt`.
By keeping track of your environment, you can minimize the impact of differences in package versions and other factors that might lead to variations in your results. This will help you achieve more reproducible results in your deep learning projects.
4. Running on the GPU also requires having the same version of CUDA and the cuDNN version to be able to reproduce the results even on the same GPU.

### Containariziation

Containers are a method of packaging and deploying applications in a way that allows them to be run consistently across different environments, such as different computers or operating systems. Containers allow developers to package an application and its dependencies, such as libraries and system tools, into a single package that can be easily deployed and run on any host with the necessary container runtime.

[Docker](https://www.docker.com) is a popular containerization platform that allows developers to build, package, and deploy applications in containers. It provides a set of tools and services for building, distributing, and running containers, including the Docker Engine, which is a runtime for executing containers, and the [Docker Hub](https://hub.docker.com), which is a cloud-based registry for storing and sharing container images.

[Singularity](https://singularity-userdoc.readthedocs.io/en/latest/) is another containerization platform that is designed specifically for use in scientific and HPC (high-performance computing) environments. It allows users to run containerized applications on clusters, grids, and clouds, and provides features such as support for MPI (Message Passing Interface) and the ability to run containers as non-root users.

One key difference between Docker and Singularity is that Singularity is designed to be more secure and to better support the needs of scientific and HPC users. For example, Singularity does not require root privileges to run containers, which makes it more suitable for use on shared computing systems where users may not have root access. Additionally, Singularity supports the use of MPI, which is a standard for communication between processes in parallel computing environments, while Docker does not. However, Docker can be run on Linux, Windows, and Mac, while Singularity is available only on Linux for now 😔.

In Part 2, I will use Docker and Singularity, as I will test it on Linux and Mac.

## Hardware-Software diffculties

TensorFlow, PyTorch, and JAX are deep learning frameworks that allow for the training of machine learning models on CPUs and GPUs. These frameworks provide a range of tools and libraries that are optimized for training on CPUs and GPUs, respectively.

GPUs are specialized hardware that can significantly accelerate the computation of certain mathematical operations that are commonly used in machine learning algorithms, such as matrix multiplications and convolutions. By using GPUs to train machine learning models, it is possible to significantly improve the performance and speed of the training process. In addition to optimizing performance, these frameworks also offer a variety of tools and libraries to make the training process easier and more efficient.

### CPU

TensorFlow, PyTorch, and JAX are all deep learning frameworks that provide various techniques and libraries to optimize the performance of machine learning algorithms on CPUs. These techniques include the use of vectorized operations, which allow for the execution of certain mathematical operations on multiple data points at once using the processor's SIMD (single instruction, multiple data) instructions, and multi-threading, which enables the parallelization of certain computations across multiple CPU cores. Additionally, all three frameworks **support** hardware acceleration libraries such as [MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-documentation.html?s=Newest) (Intel Math Kernel Library), [MKL-DNN](https://oneapi-src.github.io/oneDNN/v0/index.html) and [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page), which can **further** improve the performance of machine learning algorithms on CPUs.

In addition to these optimization techniques, PyTorch and JAX (I think that from TF2 version it also changed to that) also offer support for automatic differentiation, which allows them to compute the gradients of machine learning models during training. This can be useful for optimizing model performance and improving the efficiency of the training process. Tensorflow also changed t

It's worth noting that, while these optimization techniques can help to improve the performance of machine learning algorithms on CPUs, they can also make it more difficult to reproduce results if not properly configured. Therefore, it is important to carefully consider and document the optimization techniques used in order to ensure reproducibility in machine learning research.

### GPU
To utilize the power of GPUs with TensorFlow, PyTorch, and JAX, developers typically need to install a GPU-accelerated version of the framework as well as the necessary drivers and libraries for the GPU hardware being used. All three frameworks support the use of the CUDA and cuDNN libraries to accelerate training on GPUs, and JAX also provides support for GPU acceleration through the use of the XLA library. These libraries allow developers to harness the computing power of the GPU to accelerate the training of machine learning models and improve the performance of the training process.

Installing and configuring the necessary GPU libraries and drivers can be a complex process, and it is important to ensure that they are properly installed and configured in order to take full advantage of the GPU hardware. By using these libraries and frameworks, developers can significantly improve the performance and speed of the training process, enabling them to train more complex and powerful models in a shorter amount of time.
#### CUDA, cuDNN, cuBLAS,

You must've heard about CUDA when working with GPU. [CUDA](https://pl.wikipedia.org/wiki/CUDA) is a parallel computing platform and programming model developed by NVIDIA for general-purpose computations on graphics processing units (GPUs). It allows developers to use the power of GPUs to accelerate their applications, and it is widely used in machine learning and other fields.

[cuDNN](https://blog.roboflow.com/what-is-cudnn/) (CUDA Deep Neural Network library) is a GPU-accelerated library for deep learning developed by NVIDIA. It provides highly optimized implementations of common deep learning operations, such as convolutions and matrix multiplications, which can significantly improve the performance of deep learning models on GPUs.

[cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html) (CUDA Basic Linear Algebra Subprograms) is a GPU-accelerated library for linear algebra operations developed by NVIDIA. It provides optimized implementations of common linear algebra operations, such as matrix multiplications and decompositions, which can accelerate the performance of many machine learning algorithms.

One of the challenges of reproducibility in machine learning is the use of these GPU-accelerated libraries, which can lead to differences in the behavior of the code on different GPU hardware.

To mitigate these issues, it is important to carefully specify the GPU hardware and drivers that you are using and ensure that the same hardware and drivers are used when running the code. You can also consider using a GPU cloud service, which can provide a consistent hardware environment for running your code. However, it is worth noting that even with these precautions, there may still be some variability in the results due to differences in the underlying hardware.

#### XLA

XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear algebra developed by Google that is used in the Tensorflow machine learning framework and JAX. It is designed to optimize the performance of linear algebra operations, such as matrix multiplications and decompositions by generating efficient code for different hardware architectures, including CPUs, GPUs, and Tensor Processing Units (TPUs).

One of the benefits of using XLA is that it can improve the performance of Tensorflow models by generating highly optimized code for different hardware architectures. This can be particularly useful for large-scale machine learning tasks, where the performance of the model can have a significant impact on the time required to train and evaluate the model.

XLA is also designed to be deterministic, which can help with reproducibility in machine learning. By generating the same code for a given set of inputs, XLA can help ensure that the results of a Tensorflow and Jax model are consistent across different runs and hardware platforms. However, it is worth noting that there may still be some variability in the results due to differences in the underlying hardware and other factors.

## MLCube

As mentioned earlier, MLCube is not a new framework or service, but rather a "contract" for a consistent interface to machine learning models in containers. The "contract" is written in YAML format and specifies various details about MLCube, including the name, description, authors, platform requirements, and tasks.

The platform section of the MLCube contract allows users to specify resource requirements, such as the number of accelerators, memory and disk requirements, and other relevant information. The tasks section is where the specific steps of the MLCube are defined, including inputs, outputs, and any necessary configuration.

By following the MLCube "contract" and using the MLCube CLI, developers can easily create and train machine learning projects in a consistent and reproducible manner. You can see examples of MLCube contracts and how they are used in the MLCommons GitHub repository [examples](https://github.com/mlcommons/mlcube_examples) or in the second part of this article.

## Summary

Now that we have a basic understanding of the factors that can impact reproducibility in machine learning, it's time to put this knowledge into practice. As we have learned, reproducibility in machine learning is often hindered by the presence of randomness in various stages of the model development process, including weight initialization, data loading, and stochastic operations such as dropout. It is also important to consider the impact of different package versions, Python versions, and hardware environments on reproducibility.

While it is important to strive for reproducibility in machine learning, it is also important to recognize that it may not always be possible to achieve fully deterministic results. In practice, there may be trade-offs between reproducibility and optimization, and it is up to the developer to decide on the appropriate balance for their specific use case.

In the second part of this article, we will delve into practical examples of how to achieve reproducibility in the TensorFlow, PyTorch, and JAX frameworks when using tools such as MLCube and Docker to create consistent environments for model development and training. Let's move on to the laboratory practice phase 🔬.
