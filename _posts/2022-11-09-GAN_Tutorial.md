---
layout: post
title:  "GAN Tutorial"
date:   2022-11-12
author: Kim JeongHyeon and Khan Osama
categories: Generative Adversarial Networks
tags: GAN Tutorial
use_math: True
---

# Content
This report summarizes the tutorial presented by Ian Goodfellow at NIPS in 2016. The author answers five questions regarding generative adversarial networks (GAN) in the tutorial. These questions are:
- Why is generative modeling a topic worth studying?
- How do generative models work? Comparison of other generative models with GAN
- How do GANs work?
- What are some of the research frontiers in GANs?
- What are some state-of-the-art image models that combine GANs with other methods?
In this post, we will go through every single question and try to answer them as clearly as possible. To better grasp GANs, we modified the order of these questions.


# Internals Of GANs

Generative models refer to any model that takes a training set consisting of samples drawn from a distribution p<sub>data</sub> and learns to represent an estimate of that distribution. Here p<sub>data</sub> describes the actual distribution that our training data came from. A generative model aims to learn a distribution p<sub>model</sub>, such that this distribution is close to the actual data distribution as much as possible. Generative models are classified into two main categories; Those where we represent the p<sub>model</sub> with an explicit probability distribution and those where we do not have an explicit distribution but can sample from it. GAN belongs to the second one. 



## The GAN framework

In GANs, there are two networks, the generators, and the discriminators. The generator's job is to generate a new sample from the latent vector, which is, in turn, sampled from some prior, and the discriminator's job is to learn to distinguish the fake image from the real image. Think of the generator as a counterfeiter, trying to make fake money, and the discriminator as police, trying to separate fake money from real one. To succeed in this game, the counterfeiter must learn to make money that is indistinguishable from real money. In other words, the generator model must learn to generate data from the same distribution as the data came.
The goal of the GAN is to optimize the following equation.

<p align="center">
  <img src="/GAN_Tutorial_img/cost_function_equation.png" alt="factorio thumbnail"/>
</p>

`D(x)` represents the output from the discriminator, while `G(z)` represent the output from the generator. The first part tends to give a large negative number if the output of the discriminator for real data is not close to 1, while the second part gives a large negative number if the output of the discriminator for the fake data is not close to zero. By maximizing this term, the discriminator can successfully distinguish fake images from real ones. On the other hand, by minimizing this term, the generator can deceive the discriminator into considering the generated images as real ones. The generator can achieve this by making the output of `D(G(z))` close to 1 for fake images. This is shown below

<p align="center">
  <img src="/GAN_Tutorial_img/discriminator_vs_generator.png" alt="factorio thumbnail"/>
</p>



## Training Process

Following are the steps to train a GAN model.
1. Sample x from the training dataset and z from the prior distribution and feed them to the discriminatorn and generator, respectively.
2. Sample z from the prior distribution
3. Feed the sampled z to generator and get the generated data.
4. Feed the generated data and the real data (from step 1) to the discriminator, and get the output. 
5. Update both the discriminator network and the generator network. 
For T steps of iterations, the training process will look something like

<p align="center">
  <img src="/GAN_Tutorial_img/training_process.png" alt="factorio thumbnail"/>
</p>

The overall mechanism can be further seen in the following figure

<p align="center">
  <img src="/GAN_Tutorial_img/full_picture.png" alt="factorio thumbnail"/>
</p>


## Optimal Discriminator 

Optimizing the above term from discriminator's prospective, guarantees to reach an optimal point, only when the generator learns the ratio between the p<sub>data</sub> and p<sub>model</sub>. We can write the loss function as 

<p align="center">
  <img src="/GAN_Tutorial_img/Derivative.png" alt="factorio thumbnail"/>
</p>

The goal of the discriminator is to estimate this ratio. This is shown in the following figure
<p align="center">
  <img src="/GAN_Tutorial_img/dis_vs_gen.png" alt="factorio thumbnail"/>
</p>



## Non-Saturating Game

## GANs and Maximum Likelihood Game

## Divergence and GANs

## Tips and Tricks

# Applications Of GANs

# Research Frontiers

## Non-convergence

## Mode Collapse

## Evaluation of GANs

## Discrete Outputs

## Others



