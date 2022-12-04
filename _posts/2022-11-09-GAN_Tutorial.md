---
layout: post
title:  "GAN Tutorial"
date:   2022-11-12
author: Kim JeongHyeon, Khan Osama
categories: "GAN"
tags: Generative Adversarial Networks
use_math: True
---

# Content
This report summarizes the tutorial presented by Ian Goodfellow at NIPS in 2016. The author answers five questions regarding generative adversarial networks (GAN) in the tutorial. These questions are:
- Why is generative modeling a topic worth studying?
- How do generative models work? Comparison of other generative models with GAN
- How do GANs work?
- What are some of the research frontiers in GANs?
- What are some state-of-the-art image models that combine GANs with other methods?
<br>In this post, we will go through every single question and try to answer them as clearly as possible. To better grasp GANs, we modified the order of these questions.


# Internals of GANs

Generative models refer to any model that takes a training set consisting of samples drawn from a distribution p<sub>data</sub> and learns to represent an estimate of that distribution. Here p<sub>data</sub> describes the actual distribution that our training data comes from. A generative model aims to learn a distribution p<sub>model</sub>, such that this distribution is close to the actual data distribution as much as possible. Generative models are classified into two main categories; Those where we represent the p<sub>model</sub> with an explicit probability distribution and those where we do not have an explicit distribution but can sample from it. GAN belongs to the second one.



## The GAN framework

In GANs, there are two networks, the generators, and the discriminators. The generator's job is to generate a new sample from the latent vector, which is, in turn, sampled from some prior, and the discriminator's job is to learn to distinguish the fake image from the real image. Think of the generator as a counterfeiter, trying to make fake money, and the discriminator as police, trying to separate fake money from real one. To succeed in this game, the counterfeiter must learn to make money that is indistinguishable from real money. In other words, the generator model must learn to generate data from the distribution as the data originally came.
The goal of the GAN is to optimize the following equation.

<p align="center">
  <img src="/GAN_Tutorial_img/cost_function_equation.png" alt="factorio thumbnail"/>
  <br>
  <b> Source: [3] </b>
</p>

`D(x)` represents the output from the discriminator, while `G(z)` represents the output from the generator. The first part tends to give a large negative number if the output of the discriminator for real data is not close to 1, while the second part gives a large negative number if the output of the discriminator for the fake data is not close to zero. By maximizing this term, the discriminator can successfully distinguish fake images from real ones. On the other hand, by minimizing this term, the generator can deceive the discriminator into considering the generated images as real ones. The generator can achieve this by making the output of `D(G(z))` close to 1 for fake images. This is shown below

<p align="center">
  <img src="/GAN_Tutorial_img/discriminator_vs_generator.png" />
  <br>
  <b> Source: [3] </b>
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
  <img src="/GAN_Tutorial_img/training_process.png" />
  <br>
  <b> Source: [3] </b>
</p>

The following figure further clarify this procedure

<center>
  <p>
    <img src="/GAN_Tutorial_img/full_picture.png" />
    <br>
    <b> Discriminator (left) and Generator (right) - Source: [1] </b>
  </p>
 </center>


## Optimal Discriminator 

Optimizing the above term from discriminator's prospective, guarantees to reach an optimal point, only when the discriminator learns the ratio between the p<sub>data</sub> and p<sub>model</sub>. We can write the loss function as 

<p align="center">
  <img src="/GAN_Tutorial_img/Derivative.png" />
  <br>
  <b> Source: [3] </b>
</p>

The goal of the discriminator is to estimate this ratio. This is shown in the following figure
<p align="center">
  <img src="/GAN_Tutorial_img/dis_vs_gen.png" width="600" height="400"/><br>
  <b> Discriminator shown in dashed blue line. The goal of the discriminator is to estimate the ratio between two distribution - Source: [1] </b>
</p>

In order for the generator to align the p<sub>model</sub> distribution with the p<sub>data</sub>, the generator distribution should move towards the direction that increases the value of `D(G(z))`. This also shows that the discriminator and generator are in a cooperative rather than adversarial setting, as the discriminator finds the ratio between the distributions, and then guides the generator to climb up this ratio. 



## Non-Saturating Game

At the beginning of the training, the discriminator gets confident about the fake images quite quickly, which causes a vanishing gradient problem for the generator. A vanishing gradient means that there will be no update for the generator, even for the bad samples. To fix this problem, we can change the equation for the generator from `log(1-D(G(z)))` to `-log(D(G(z)))`. Considering a sigmoid function at the end of the discriminator network, we can see that the gradient is equal to zero at the beginning of the training. Modifying the sign reverse this phenomenon and brings the vanishing gradient problem for the real data. However, this is okay for real data, as this gradient is not used to update the generator. This is show below
<p align="center">
  <img src="/GAN_Tutorial_img/non_saturating.png" /> <br>
    <b> The left figure shows the output of the sigmoid function and its gradient. The final layer of the discriminator is sigmoid function. The middle figure shows the default settings and the gradient under this default setting. Finally, the right most figure shows the solution to the vanishing gradient - Source: [2] </b>

</p>

## GANs and Maximum Likelihood Game


GANs models are capable of doing maximum likelhood learning. However, to achieve this, we need to minimize the KL-divergence instead of JSD between the two distributions. This can be achieved by finding such a loss function for generator whose derivative will be equal to the derivative of KL divergence.  


## Divergence and GANs

GAN minimizes the Jensen-Shannon divergence(JSD) between the p<sub>data</sub> and p<sub>model</sub>. To prove it, we will continue the derivation from the optimal discriminator part. 

<p align="center">
  <img src="/GAN_Tutorial_img/jsd.png" alt="factorio thumbnail"/>
  <br>
  <b> Source: [3] </b>
</p>

We can see that minimizing the main loss function of GANs, indeed, minimizes the JSD between the two distributions. Alternatively, we can also modify the generative models to minimize the Kullback–Leibler divergence (KL) between the two distributions. This would allow the GAN to do maximum likeliood learning. In order to do so, we need to change the loss function for the generator. Specifically, changing the loss function for the generator to the following loss function will allow the GAN to minimize the KL divergence.

<p align="center">
  <img src="/GAN_Tutorial_img/new_cost_mle.png" alt="factorio thumbnail"/>
    <br>
  <b> Source: [3] </b>
</p>


 Different setting results in different optimization. KL-divergence tries to fit the p<sub>model</sub> to all the peaks of the p<sub>data</sub>, and therefore average out over all the modes. On the other hand, JSD tries to fit the p<sub>model</sub> to a single peak or model. This is shown in the following figure.

<p align="center">
  <img src="/GAN_Tutorial_img/divergences.png" alt="factorio thumbnail"/><br>
  <b> The choice of divergence defines the optimization behavior - Source: [2] </b>

</p>

The fact that GANs try to fit p<sub>model</sub> to a single mode rather than averaging over multiple modes might be an explanation for why GANs produce good-quality images. However, recent efforts have shown that high quality images can also be produced by GAN when doing maximum likelihood learning.

## Tips and Tricks
There are several tricks that we can use for training GANs. Some of the tricks are shown below:

Using labels to train a GAN network can improve the quality of samples generated by the model. The specific reason why providing labels works still needs to be clarified. Nonetheless, doing so improves the quality of the samples and makes them closer to the one our eyes expect the most.

If the discriminator depends only on a small sets of features to identify the real images, then the generator can easily deceive the discriminator by producing samples that have those sets of features only. This makes the discriminator overconfident for samples with specific sets of features only, even if those samples make no sense. To avoid this situation, we allow the discriminator's output to be between 0 and 0.9. This way, even if the discriminator is sure about an image, there will be gradient for the generator to learn from. In Tensorflow, this loss can be modified as below
<p align="center">
  <img src="/GAN_Tutorial_img/loss_function.png" alt="factorio thumbnail"/>
  <br>
  <b> Source: [3] </b>
</p>
Smoothing the labels for the fake samples will generate unexpected behaviors. 

Batch Normalization (BN) creates a dependency between the samples in a batch when the size of the batch is quite small. This is problematic in GANs because when the size of the batch gets too small, the normalization constants in BN starts fluctuating. This will make the model dependent on these fluctuating constants rather than the input noise. The following figure shows this phenomenon
<p align="center">
  <img src="/GAN_Tutorial_img/batch_norm.png" alt="factorio thumbnail" width="500" height="500"/> <br>
  <b> Dependence between different images in a same batch - Source:[1] </b>

</p>

The generated images in the same batch (Two batches, top and down) are similar. Virtual batch normalization avoids this problem by sampling a reference batch before training and finding this batch's normalization parameters. In the subsequent training steps, these normalization parameters are used together with the current batch to recompute the normalization parameters and use them during the training process.


# Applications Of GANs
Generative models are of great use in real-life. Some of the examples are as follows:
1. They provide a way to represent and manipulate high-dimensional probability distributions, which are quite useful in applied math and engineering.
2. Reinforcement learning depends on the feedback they get from their environment. For efficiency and safety reasons, it is better to use a simulated environment than an actual one. Generative models can be used to generate the environment for the agent. 
3. Generative models can be used with semi-supervised learning, in which the labels for most of the data are missing. Given that semi-supervised learning can be either transductive or inductive learning, the generator models can serve as a good transductive part.
4. For a given input, sometimes it is desirable to have multiple outputs. The existing approach uses MLE to average out all the possible outputs, resulting in poor results from the model. The generative model can be used to put focus on one of many possible outputs.
5. Generative models are achieving state-of-the-art performance in recovering low-quality images. The knowledge of how the generative models learn actual high-resolution images is used to recover these low-quality images. 
Some of the other applications are image-to-image translation, text-to-image translation, image-to-text translation, and many creative projects where the goal is to create art. 

The applications of generative models are not restricted to the above-mentioned ones. With more and more research coming out in this area every day, new ways are being invented to embed these models in our daily life.

# Research Frontiers

Back when this paper was published, GANs were relatively new and had many research oppurtunities.

## Non-convergence

The nature of the GAN settings is such that the two networks compete with each other. In simple words, one network maximizes a value while the other network minimizes the same value. This is also known as a zero-sum non-cooperative game. In game theory, GAN converges when both networks reach nash equilibrium. In nash equilibrium, one network's actions will not affect the course of the other network's actions. Consider the following optimization problem: `minmax V(G,D) = xy` 

The nash equilibrium of this state reaches when `x=y=0`. The following figure shows the result of gradient descent on the above function. 
<p align="center">
  <img src="/GAN_Tutorial_img/oscillations.png" width="200" height="300"/><br>
  <b> Optimization in game theory can result in sub-optimal structure - Source: [4]</b>
</p>

This clearly shows that some cost functions might not converge using gradient descent.

### Mode Collapse

In reality, our data has multiple modes in the distribution, known as multi-modal distributions. However, sometimes, the network can only consider some of these modes when generating images. This gives rise to the problem called model collapse. In model collapse, only a few modes of data are generated. 

Basically, we have two options to optimize the objective function for the GANs. One is <b>min<sub>G</sub>max<sub>D</sub> V(G,D)</b> while the other is <b>max<sub>D</sub>min<sub>G</sub> V(G,D)</b>

They are different, and optimizing them corresponds to optimizing two different functions. 

In the `maxmin` game, the generator minimizes the cost function first. It does this by mapping all the values of `z` to a particular `x`, which can be used to deceive the discriminator. And hence generator will not be learning useful mapping. On the other hand, in the `minmax` game, we first allow the discriminator to learn and then guide the generator to find the modes of the underlying data. 

What we want the network to do is `minmax`; however, since we update the networks simultaneously, we end up performing `maxmin`. This gives rise to the mode collapse. The following figure shows this behavior

<p align="center">
  <img src="/GAN_Tutorial_img/model_collapse.png" /><br>
  <b> Mode collapse in toy dataset - Source: [1] </b>
</p>

The generator visits one mode after another instead of learning to visit all different modes. The generator will identify some modes that the discriminator believes are highly likely and place all of its mass there, and then the discriminator will learn not to be fooled by going to only a single mode. Instead of the generator learning to use multiple modes, the generator will switch to a different mode, and this cycle goes on. The following note from google machine learning website best explains the cause:

<p align="center">
"If the generator starts producing the same output (or a small set of outputs) over and over again, the discriminator's best strategy is to learn to always reject that output. But if the next generation of discriminator gets stuck in a local minimum and doesn't find the best strategy, then it's too easy for the next generator iteration to find the most plausible output for the current discriminator. Each iteration of generator over-optimizes for a particular discriminator, and the discriminator never manages to learn its way out of the trap. As a result the generators rotate through a small set of output types" <br><b>Source: <a href="https://developers.google.com/machine-learning/gan/problems">Google Machine Learning</a> </b> 
</p>

Two common methods to mitigate this problem are minibatch features and unrolled GANs. 

In minibatch discrimination, we feed real images and generated images into the discriminator separately in different batches and compute the similarity of the image x with images in the same batch. We append this similarity to one of the layers in the discriminator. If the model starts to collapse, the similarity of generated images increases. This is a hint for the discriminator to use this score and penalize the generator for putting a lot of mass in one region. 

Initially, when we update the networks simultaneously, we do not consider the maximized value of the discriminator for the generator. In unrolled GANs, we can train the discriminator for `k` steps and build the graph for each of these steps. Finally, we can propagate through all these steps and update the generator. By doing so, we can update the generator not only with respect to the loss but also with respect to the discriminator's response to these losses. This is proved to be helpful in mode-collapse problems, as shown below.

<p align="center">
  <img src="/GAN_Tutorial_img/mode_solved.png"  /><br>
  <b> Urolled GAN solved the problem of mode collapse in toy dataset - Source [1] </b>
</p>

You can think of unrolled GANs as a way for generator to see in the future and find out which direction the discriminator is taking it. This will help the generator not to focus on one discriminator only.


## Conclusion

1. GANs are type of generative models which is based upon the game theory. Specifically, in GANs, two networks compete against each other.
2. GANs use supervised ratio estimation technique to approximate many cost functions, including the KL divergence used for maximum likelihood estimation.
2. Training GANs require Nash equilibrium which is high dimentional, continuous, non-convex games. 
4. GANs are crucial to many state of the art image generation and manipulation systems and have many potentials in the future.  


**References**

[1] <a href="https://arxiv.org/pdf/1701.00160.pdf"> Original Paper </a>

[2] <a href="https://sites.google.com/view/berkeley-cs294-158-sp20/home"> CS294 </a>

[3] <a href="https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2020"> Deep Learning for Computer Vision </a>

[4] <a href="https://jonathan-hui.medium.com/gan-why-it-is-so-hard-to-train-generative-advisory-networks-819a86b3750b#4987"> Medium post - Training in GANs </a>

[5] <a href="https://jonathan-hui.medium.com/gan-whats-generative-adversarial-networks-and-its-application-f39ed278ef09"> Medium post - GANs </a>


