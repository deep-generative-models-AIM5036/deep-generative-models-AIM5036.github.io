---
layout: post
title:  "Markov Chain Monte Carlo and Variational Inference: Bridging the Gap."
date:   2022-11-13
author: Sungmin Kang
categories: Latent-Variable
tags: VI latent-variable MCMC
use_math: true
---
- Tim Salimans, Diederik P. Kingma, and Max Welling. Markov Chain Monte Carlo and Variational Inference: Bridging the Gap. Proceedings of the 32nd International Conference on Machine Learning, 2015


## Introduction
Markov Chain Monte Carlo(MCMC)는 보다 정확한 모델을 계산할 수 있지만 샘플링에 시간이 매우 오래 걸리는 단점이 있고, Variational Inference(VI)는 드는 시간을 단축시켰지만 모델의 표현력이 떨어지는 문제가 존재합니다.
본 논문 Markov Chain Monte Carlo and Variational Inference: Bridging the Gap(이하 MCMC and VI)에서는 두 개의 간극을 메우는 방법을 제시하고, 이 방법의 실제 구현으로 Hamiltonian VI 구현체를 제시합니다.

### Markov Chain Monte Carlo
Variational Inference는 강의 내용에서도, 다른 논문의 포스팅에서도 많이 다루어졌으므로 내용은 생략하고, Markov Chain Monte Carlo에 대해 간략하게 설명하고 넘어가도록 하겠습니다.

#### Monte Carlo
무작위로 매우 많은 시행을 할 경우, 실제 값에 근사한 결과가 나오는 것을 활용합니다.
이 예시로 흔히 보신 것은 -1~1 사이에 점을 무수히 많이 찍어서, 원 범위 안에 있는 점의 개수를 구한 뒤, 원주율을 근사하는 것이 있을 것 같습니다.
실제로 적절한 무작위 생성이 있다면 시행 횟수가 늘어남에 따라 실제 원주율 값에 근사하는 것을 보실 수 있습니다.

<p align="center"><img src="/assets/MCMCandVI/monte_carlo.png" width=50%></p>


#### Markov Chain
Markov Chain은 이전의 상태만을 바탕으로 현재의 상태에 영향이 가도록 구성된 체인입니다.
간단한 예시를 들자면, 어제 비가 왔을 때 오늘 비가 올 확률과 오지 않을 확률이 있고 - 어제 비가 오지 않았을 때 오늘 비가 올 확률과 오지 않을 확률이 제시된 상황을 생각해보시면 됩니다.

<p align="center"><img src="/assets/MCMCandVI/markov_chain.png" width=40%></p>

#### Monte Carlo + Markov Chain
둘을 결합한 Markov Chain Monte Carlo는, Markov Chain에서의 확률을 바탕으로 실제로 무작위 샘플링을 통해 시뮬레이션하는 Monte Carlo식으로 하는 것입니다.
즉, 이전 상태값을 활용한 무작위 동작을 무한히 반복한다면, 실제 분포와 근사한 분포를 얻을 수 있다는 것입니다.

'이 이전 상태값을 활용한 무작위 동작'은 여러가지 방식이 있습니다. 간단한 예시인 Metropolis 알고리즘을 예로 들면, 현재 위치에서 정규분포상의 무작위 값을 바탕으로 근처로 이동을 시도합니다. 이동했을 때의 가능성이 더 높다면 실제로 이동하고, 그렇지 않다면 거절합니다. 다만 때때로 거절할 상황에도 수용하는 상황이 생깁니다.

<p align="center" width="100%">
    <img width="50%" src="/assets/MCMCandVI/mcmc_move.png">
</p>

왼쪽은 더 가능성이 높아 이동한 경우, 중앙은 가능성이 더 낮아져 이동하지 않은 경우, 오른쪽은 가능성이 낮아지는 경우임에도 수용한 경우입니다.
이러한 동작을 반복하면 실제 분포를 얻어낼 수 있게 됩니다.

<p align="center" width="100%">
    <img width="50%" src="/assets/MCMCandVI/mcmc_dist.gif">
</p>

위 시뮬레이션은 [https://chi-feng.github.io/mcmc-demo/app.html](https://chi-feng.github.io/mcmc-demo/app.html) 에서 직접 수행해보실 수 있습니다.

 
### This paper

#### Background
위에서 설명드린 내용을 바탕으로, MCMC는 점근적으로(asymptotically) 정확해집니다. 
다만 이 과정에서 극도로 많은 시뮬레이션이 필요할 것이므로 느립니다.
더불어 좋은 Markov Chain을 구성하는 데에 어려움이 있고, 값이 실제에 근사했는지를 확인하는 것조차 어렵습니다.

이에 반해 VI는 정확한 분포를 예측할 수 있다는 사실 자체를 보장할 수 없는 문제가 있지만 더 빠르게 수행할 수 있습니다.

#### Contribution
본 논문은 MCMC가 실제 분포에 근사할 수 있다는 가능성과, VI의 낮은 비용을 가져오는 것을 제안합니다.
더불어 이의 실제 구현체인 Hamiltonian VI를 Auxiliary variable과 Stochastic Gradent VI를 활용하여 만들었습니다.


## MCMC, more details
앞서 설명드렸던 Metropolis의 일부분에서, 단순히 정규분포 값을 활용해서 이동한다면 Local Optima에 갇히는 현상이 발생하게 됩니다.
이를 해결하기 위해 일정 수준 이하로 가능성이 낮아지는 경우 수용하도록 한 것이 Metropolis 알고리즘입니다.
다만 이러한 경우에도 Local Optima 근처에서 맴도는 현상이 종종 발생하는 단점은 존재합니다.

이러한 MCMC의 단점을 해결하는 것은 더 좋은 Markov Chain을 구성하는 것, 즉 더 나은 샘플링 기법을 고안하는 것입니다.
이를 위해서는 본 논문에 제시된 Gibbs Sampling, Over-relaxation 등을 포함하여 Hamiltonian MC(HMC) 등의 많은 기법이 연구되고 있으며 Jiaming Song (2017) A-NICE-MC: Adversarial Training for MCMC 과 같은 관련 논문도 제시되고 있습니다.


## MCMC and Auxiliary Variables
논문 내용의 시작은 VI에서 시작합니다.
VI에서 얻어온 식 (1)과 (2) 두개를 바탕으로,

$$\begin{align}
\log {p(x)} &\geq \log{p(x)} - D_{KL}(q_\theta(z|x)||p(z|x)) \\ 
&= \Bbb{E}_{q_\theta(z|x)}[\log{p(x,z)}-\log{q_\theta(z|x)}]=\mathcal{L}.
\end{align}$$

ㅇㅇ

$W$ ㅇㅇ
