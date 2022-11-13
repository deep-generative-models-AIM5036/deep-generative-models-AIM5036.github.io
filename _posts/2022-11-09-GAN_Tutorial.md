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

In order for the generator to align the p<sub>model</sub> distribution with the p<sub>data</sub>, the generator value should move towards the direction that increases the value of `D(G(z))`. This also shows that the discriminator and generator are in a cooperative rather than adversarial setting, as the discriminator finds the ratio between the distributions, and then guides the generator to climb up this ratio. 



## Non-Saturating Game

At the beginning of the training, the discriminator gets confident about the fake images quite quickly, which causes a vanishing gradient problem for the generator. A vanishing gradient means that there will be no update for the generator, even for the bad samples. To fix this problem, we can change the equation for the generator from `log(1-D(G(z)))` to `-log(G(z))`. Considering a sigmoid function at the end of the discriminator network, we can see that the gradient is equal to zero at the beginning of the training. Modifying the sign reverse this phenomenon and brings the vanishing gradient problem for the real data. However, this is okay for real images, as this gradient is not used for updating the generator.This is show below
<p align="center">
  <img src="/GAN_Tutorial_img/non_saturating.png" alt="factorio thumbnail"/>
</p>

## GANs and Maximum Likelihood Game


GANs models are capable of doing maximum likelhood learning. However, to achieve this we now need to minimize the KL-divergence between the two distributions instead of JSD. This can be achieved by finding the loss function for generator whose derivative will be equal to the derivative of KL divergence.  


## Divergence and GANs

GAN minimizes the Jensen-Shannon divergence(JSD) between the p<sub>data</sub> and p<sub>model</sub>. To prove it, we will continue the derivation from the optimal discriminator part. 

<p align="center">
  <img src="/GAN_Tutorial_img/jsd.png" alt="factorio thumbnail"/>
</p>

We can see that minimizing the main loss function of GANs, indeed, minimizes the JSD between the two distributions. Alternatively, we can also modify the generative models to minimize the Kullback–Leibler divergence (KL) between the two distributions. This would allow the GAN to do maximum likeliood learning. In order to do so, we need to change the loss function for the generator. Specifically, changing the loss function for the generator to the following loss function will allow the GAN to minimize the KL divergence.

<p align="center">
  <img src="/GAN_Tutorial_img/new_cost_mle" alt="factorio thumbnail"/>
</p>


 Different setting results in different optimization. KL-divergence tries to fit the p<sub>model</sub> to all the peaks of the p<sub>data</sub>, and therefore average out over all the modes. On the other hand, JSD tries to fit the p<sub>model</sub> to a single peak or model. This is shown in the following figure.

<p align="center">
  <img src="/GAN_Tutorial_img/divergences.png" alt="factorio thumbnail"/>
</p>

The fact that GANs try to fit p<sub>model</sub> to a single mode rather than averaging over multiple modes might be an explanation for why GANs produce good-quality images. However, recent efforts have shown that high quality images can also be produced by GAN when doing maximum likelihood learning.

## Tips and Tricks
There are several tricks that we can use for training GANs. Some of the tricks are shown below:
1. Using labels to train a GAN network can improve the quality of samples generated by the model. The specific reason as to why providing labels work is still not clear. Still, doing so improves the quality of the samples and make it more close to the one that our eyes expect.
2. If discriminator depends only on small set of features to identify the real images, then the generator can easily deceive the discriminator by producing the samples that have those set of features only. This makes the discriminator overconfident for samples that have those specific set of features only, even if the samples make no sense. To avoid this situation, we allow the discriminators output to be between 0 and 0.9. This way, even if the discriminator is sure about an image, there will be gradient for the generator to learn from. In Tensorflow, this loss can be modified as below
<p align="center">
  <img src="/GAN_Tutorial_img/loss_function.png" alt="factorio thumbnail"/>
</p>
Smoothing the lables for the fake samples, will generate unexpected behaviours.
3. Batch Normalization creates a dependency between the samples in a batch. This results in generated images that are not independent of each other. The following figure shows this phenomenon
<p align="center">
  <img src="/GAN_Tutorial_img/batch_norm.png" alt="factorio thumbnail"/>
</p>
We can see that the generated images in the same batch (Two batches, top and down) are somewhat similar. To avoid this problem, virtual batch normalization samples a reference batch before training and finds the normalization parameters of this batch. In the subsequent training steps, this normalization parameters are used together with the current batch to recompute the normalization parameters and use them during the training process.


# Applications Of GANs

# Research Frontiers

## Non-convergence

## Mode Collapse

## Evaluation of GANs

## Discrete Outputs

## Others



