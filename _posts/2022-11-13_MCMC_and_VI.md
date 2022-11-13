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
저희가 계산해야 할 부분은 모델이 출력한 logistic 분포의 파라미터로 정답인 x픽셀 값을 예측할 확률을 구하는 것입니다. 그럼 모델이 출력한 분포값이 나올 부분을 적분하면 됩니다. 적분을 구하기 위해 저희는 logistic 분포의 CDF인 sigmoid를 사용합니다.

식(1)은 mixture of logistic distribution식이다. pi로 각 logistic distribution의 기여도를 결정하고, 각 logistic 분포의 파라미터 mu와 scale이 존재합니다. 식(2)를 이해하기 위해서 그래프를 가져와봤습니다. 더 심플하게 설명하기 위해 mixture가 아닌 simple logistic으로 설명해보겠습니다. 윗 그래프는 분포로 Logistic PDF이고 아래는 각자 해당되는 CDF이고 (variance를 scale로 해석해주시면 됩니다), CDF+과 CDF-는 픽셀 구간(PixelCNN 같은 경우 +0.5와 -0.5)입니다.보시다시피 빨간색 분포에서 CDF+와 CDF-이 CDF 함수를 만나는 지점들이 간격이 제일 큽니다. 확률적으로 해석하면 0에 샘플링될 확률이 셋 중에서 제일 높다는 뜻입니다. 그럼 이젠 식(2)를 둘러보겠습니다. 보시다시피 X = mu = 0일 때 빨간 그래프 상황이 이루어지고 그럴 경우에 P(X)가 극대화됩니다. P(x)에 -log만 추가해주면 로스 함수로 만들고 학습이 가능합니다. 이와 함께 분포의 weight값 pi도 학습하게 됩니다.

<p align="center" width="100%">
    <img width="50%" src="/assets/PixelCNN++_img/gaussian_pdf_cdf.png">
</p>

 Discretized Logistic Mixture Likelihood을 사용한 점이 이 논문의 핵심이 됩니다. 저희가 모델링하는 대상 이미지는 natural하고 continuous한 분포를 따르는데, softmax을 사용했을 때 그런 분포를 학습하는 게 어렵습니다. 아래 분포는 모두 랜덤 픽셀값 생성 분포이고, 상단에는 softmax로 학습한 것이고 하단은 mixture of logistic distributions을 사용했습니다. Logistic distribution을 사용한 버전이 실제 데이터 분포를 더 반영하고 있다는 것이 바로 확인됩니다.
 
<p align="center" width="100%">
    <img width="50%" src="/assets/PixelCNN++_img/softvsmoldist.png">
</p>

#### Conditioning on Whole Pixels

기존 PixelCNN에서 각 RGB값을 생성할 때 autoregressive하게 factorized하게 생성합니다. PixelCNN++에서는 RGB를 autoregressive하게 생성하지만 모두 동일한 feature map을 사용하고 서로 linear한 관계를 갖게 설정했습니다. 그 뜻은, green 과 red, 또는 blue와 green,red의 관계를 coefficient로 표현이 가능하다는 것입니다. 그럼 모델이 실질적으로 출력하는 값들은 mixture logistic distribution의 파라미터들과 linear coefficients $\alpha$ , $\beta$ 와 $\gamma$ 입니다.

<p align="center" width="100%">
    <img width="65%" src="/assets/PixelCNN++_img/conditioning_on_whole_pixels.png">
</p>

#### Downsampling Versus Dilated Convolution

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
