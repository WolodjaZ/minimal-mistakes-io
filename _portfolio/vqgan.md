---
title: "VQGAN JAX/Flax"
excerpt: "Implementation of VQGAN in JAX/Flax"
data: 2022-11-24
languages: [python, jax, flax]
tags: [VQGAN, JAX, Flax]
thumb: /images/thumbs/portfolio/vqgan.png
---

This is just a short explanation, so please refer to the [documentation](https://wolodjaz.github.io/jax-vqgan/) and [github repo](https://github.com/WolodjaZ/jax-vqgan).

This projects implement the VQGAN architecture in JAX/Flax. So what are these names?

[JAX](https://jax.readthedocs.io/en/latest/index.html) (Just After eXecution) is a recent machine/deep learning library developed by DeepMind and Google. Unlike Tensorflow, JAX is not an official Google product and is used for research purposes. The use of JAX is growing among the research community due to some really cool features. Additionally, the need to learn new syntax to use JAX is reduced by its NumPy-like syntax.

[Flax](https://flax.readthedocs.io/en/latest/index.html) is a high-performance neural network library for JAX that is designed for flexibility: Try new forms of training by forking an example and by modifying the training loop, not by adding features to a framework.

VQGAN (Vector Quantized Generative Adversarial Network): VQGAN is a GAN architecture which can be used to learn and generate novel images based on previously seen data. It was first introduced for the paper [`Taming Transformers`](https://arxiv.org/abs/2012.09841) (2021). It works by first having image data directly input to a GAN to encode the feature map of the visual parts of the images. This image data is then vector quantized: a form of signal processing which encodes groupings of vectors into clusters accessible by a representative vector marking the centroid called a “codeword.” Once encoded, the vector quantized data is recorded as a dictionary of codewords, also known as a codebook. The codebook acts as an intermediate representation of the image data, which is then input as a sequence to a transformer. The transformer is then trained to model the composition of these encoded sequences as high resolution images as a generator.

![VQGAN](https://raw.githubusercontent.com/CompVis/taming-transformers/master/assets/teaser.png)