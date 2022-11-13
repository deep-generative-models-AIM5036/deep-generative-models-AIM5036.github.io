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

1. Introduction
2. Modifications
3. oDiscretized Logistic Mixture Likelihood
4. Conditioning on Whole Pixels
5. Downsampling Versus Dilated Convolution
6. Adding Short-cut Connections
7. Regularization Using Dropout
8. Unconditional Generation (CIFAR10)
9. From PixelCNN to PixelSNAIL
10. Concusion

## Introduction
Markov Chain Monte Carlo(MCMC)는 보다 정확한 모델을 계산할 수 있지만 샘플링에 시간이 매우 오래 걸리는 단점이 있고, Variational Inference(VI)는 드는 시간을 단축시켰지만 모델의 표현력이 떨어지는 문제가 존재합니다.
본 논문 Markov Chain Monte Carlo and Variational Inference: Bridging the Gap(이하 MCMC and VI)에서는 두 개의 간극을 메우는 방법을 제시하고, 이 방법의 실제 구현으로 Hamiltonian VI 구현체를 제시합니다.

### Markov Chain Monte Carlo
Variational Inference는 강의 내용에서도, 다른 논문의 포스팅에서도 많이 다루어졌으므로 내용은 생략하고, Markov Chain Monte Carlo에 대해 간략하게 설명하고 넘어가도록 하겠습니다.

#### Monte Carlo
무작위로 매우 많은 시행을 할 경우, 실제 값에 근사한 결과가 나오는 것을 활용합니다.
이 예시로 흔히 보신 것은 -1~1 사이에 점을 무수히 많이 찍어서, 원 범위 안에 있는 점의 개수를 구한 뒤, 원주율을 근사하는 것이 있을 것 같습니다.
실제로 적절한 무작위 생성이 있다면 시행 횟수가 늘어남에 따라 실제 원주율 값에 근사하는 것을 보실 수 있습니다.

<p align="center" width="100%">
    <img width="65%" src="/assets/MCMCandVI/monte_carlo.png">
</p>

#### Markov Chain
Markov Chain은 이전의 상태만을 바탕으로 현재의 상태에 영향이 가도록 구성된 체인입니다.
간단한 예시를 들자면, 어제 비가 왔을 때 오늘 비가 올 확률과 오지 않을 확률이 있고 - 어제 비가 오지 않았을 때 오늘 비가 올 확률과 오지 않을 확률이 제시된 상황을 생각해보시면 됩니다.

<p align="center" width="100%">
    <img width="65%" src="/assets/MCMCandVI/markov_chain.png">
</p>

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
    <img width="50%" src="/assets/MCMCandVI/mcmc_dist.jpg">
</p>

위 시뮬레이션은 (https://chi-feng.github.io/mcmc-demo/app.html
)[https://chi-feng.github.io/mcmc-demo/app.html] 에서 직접 수행해보실 수 있습니다.

 
<p align="center" width="100%">
    <img width="50%" src="/assets/PixelCNN++_img/softvsmoldist.png">
</p>

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

기존 PixelCNN에서는 비교적 receptive field이 작은 convolution을 사용합니다. long dependency 관계를 포착하기 위해 인풋을 dilated convolution으로 압축하면서 receptive field를 늘립니다 (그런 후 feature map을 다시 spatially 키워줍니다). 하지만 computation cost 측면에서 convolution의 stride를 키워주면서 인풋을 압축하는 게 더 유리합니다. 따라서 여기는 dilated convolution을 사용하지 않고 stride가 더 높은 convolution을 사용합니다.                   

<p align="center" width="100%">
    <img width="50%" src="/assets/PixelCNN++_img/stride.png">
</p>

#### Adding Short-cut Connections

Stride가 높을수록 정보 손실이 일어날 수 있습니다. 이 점을 보완하기 위해서 short-cut connection을 사용합니다 (ResNet layer 1과 6, 2와 5, 3과 4에 short-cut connection).

<p align="center" width="100%">
    <img width="65%" src="/assets/PixelCNN++_img/architecture.png">
</p>

Short-cut connection을 적용한 것과 비교했을 때 NLL가 다음과 같습니다. 

<p align="center" width="100%">
    <img width="65%" src="/assets/PixelCNN++_img/shortcut.png">
</p>

#### Regularization Using Dropout

PixelCNN는 충분히 overfit 할 capacity를 갖고 있습니다. 따라서 dropout으로 regularize 해줍니다. 실제로 dropout을 사용하지 않았을 때 training set에서 NLL가 2.0 bits per-sub-pixel로 측정이 되는데 test set에서는 6.0을 넘는다. 아래 이미지는 dropout 없이 학습하고 생성한 것이다.

<p align="center" width="100%">
    <img width="30%" src="/assets/PixelCNN++_img/dropout.png">
</p>

#### Unconditional Generation (CIFAR10)

PixelRNN는 NLL가 우수한 대신 느리다는 단점이 있고, 반면에 PixelCNN는 빠르지만, PixelRNN보다 NLL이 높습니다. PixelCNN의 문제점들을 해결한 후, PixelCNN++는 SOTA인 2.92 bits per sub-pixel를 얻었습니다 (CIFAR10). 

<p align="center" width="100%">
    <img width="40%" src="/assets/PixelCNN++_img/result.png">
</p>

#### From PixelCNN to PixelSNAIL

그럼 실제로 PixelCNN의 receptive field는 어떻게 작동될까요?
아래 그림을 보시면 중앙 픽셀에 생성에 있어서 전 픽셀들의 영향을 계산합니다. Random initialization에서 각 픽셀의 gradient를 계산하고 값이 0.001보다 클 경우 칠해줍니다. 보시다시피 PixelCNN 비해 PixelCNN++가 더 큰 receptive field를 갖고 있습니다. 아마 short-cut connection때문에 그러지 않을까 예상합니다. 하지만 둘 다 여전히 이전 픽셀들 전체를 고려하는게 아니라는게 단점입니다. 이걸 해결하기 위헤 PixelSNAIL는 attention block을 residual block과 함께 사용해서 long-dependency 관계도 고려하게 됩니다.

<p align="center" width="100%">
    <img width="75%" src="/assets/PixelCNN++_img/receptivefield.png">
</p>

#### Conclusion

PixeCNN++의 문제 설정과 문제 해결법은 간단합니다. PixelCNN 구조로 더 뛰어난 성능을 보여줄 수 있다고 가설을 세웠고, 개선점들을 적용한 후 실제로 성능이 증가했다는 것을 보여줬습니다. 현재 PixelCNN/PixelCNN의 대표적 application은 latent 분포를 학습하는 데에 있습니다. 예를 들어 VQVAE의 latent 분포는 지정된 prior 분포가 아니고, 그 분포를 PixelCNN로 학습하게 되면 Encoder 없이 랜덤 생성이 가능합니다.
