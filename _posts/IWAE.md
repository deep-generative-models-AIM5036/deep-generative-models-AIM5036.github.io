---
layout: post
title:  "IWAE"
date:   2022-11-11
author: Jongwon Park, Kyeongrok Park
categories: Flow
tags: IWAE, Flow
use_math: true
---

# **IMPORTANCE WEIGHTED AUTOENCODERS(IWAE)**

## Yuri Burda, Roger Grosse, Ruslan Salakhutdinov [ICLR 2015]
-------------

본 포스팅은 IWAE(IMPORTANCE WEIGHTED AUTOENCODERS)논문을 정리한 문서입니다.

## 1. Preliminary: What is Latent Variable Model?   

IWAE의 핵심 아이디어를 보기 전에 백드라운드로 latent variable model (LVM)에 대해 알아보겠다.
그림 1은 가장 심플한 LVM의 예시를 보여준다. 그림에서 x는 우리가 실제로 볼 수 있는 데이터 샘플이고, z는 우리가 볼 수 없지만 x를 생성하는데 사용되는 분포다. 
 
![Figure1](/assets/IWAE_img/Figure1.jpg) (그림 1 Latent Variable Model)

z는 하나의 known distribution으로 정할 수도 있고, 학습을 통해 배울 수도 있는데, VAE에서는 정해 놓는 방식을 사용한다. 대표적으로 데이터가 discrete하면 bernoulli, continuous하면 gaussian 분포를 사용한다.
만일 z를 Bernoulli로 설정했다면, 아래와 같은 식을 통해 $p_{theta}(x|z)$를 구할 수 있다. 

$z = (z_{1},z_{2},...,z_{k}) ~ p(z;\beta) = \prod_{k=1}^{K} \beta_{k}^{z_{k}}(1-\beta_{k})^{1-z_{k}}$ (식 1)
$x = (x_{1},x_{2},...,x_{k}) ~ p(x|z) = Bernoulli(x_{i}; DNN(z))$ (식 2)

z가 deep neural network와 sigmoid함수를 사용하여 0~1값을 준다면, x는 해당 값을 가지고 biased coin flipping을 하듯이 샘플링 할 수 있다. 샘플에 대한 likelihood evaluation은 다음과 같다.

$p_{theta}(x) = \sum_{z} P_{z}(z)p_{\theta}(x|z)$ (식 3)

식 3은 특정 x에 대해서, 모든 underlying cause(z)가 주어졌을 때 해당 x가 나올 확률들을 모두 더한다. $\sum$ 기호를 풀어보면 이해가 쉬울 것이다. z가 1부터 k까지의 값을 가질 수 있을 때, $p_{\theta}(x)$는 $p_{z}(z1)p_{\theta}(x|z1)+p_{z}(z2)p_{\theta}(x|z2)+...+p_{z}(zk)p_{\theta}(x|zk)$로 나타낼 수 있다.

Training은 N번 샘플된 x값들을 이용해서 likelihood evaluation (3번식)을 maximize하는 $\theta$값을 찾는 과정이다. Train objective는 다음과 같이 표현 할 수 있다.
$max_{\theta}\sum_{i}^{N}logp_{\theta}(x^(i)) = \sum_{i}^{N}log \sum_{z}^{K}p_{z}(z)p_{\theta}(x^(i)|z)$ (식 4)
**이러한 과정은 K값이 작다면 아무런 문제가 되지 않는다. **

만일 K값이 한정되어 있다면, 우리는 exact한 training objective (식 4)를 얻을 수 있다. 예를 들어 $p_{theta}(x|z)$가 mixture of 3 gaussians이고 $p_{z}(z)$가 uniform distribution이라고 해보겠다.
$p_{theta}(x|z=k) = \frac{1}{(2\pi)^{\frac{n}{2}} |\sum_{k}|^\frac{1}{2}} exp(-\frac{1}{2}(x-\mu_{k})^{T}\sum_{k}^{-1}(x-\mu_{k}))$ 이면서 $p_{z}(z=A)=p_{z}(z=B)=p_{z}(z=C)$이기 때문에 training objective는 다음과 같다. 

$max_{\theta}\sum_{i}logp_{\theta}(x^{(i)}) = max_{\mu, \sigma}\sum_{i}log[ \frac{1}{3}\frac{1}{(2\pi)^{\frac{n}{2}} |\sum_{A}|^\frac{1}{2}} exp(-\frac{1}{2}(x-\mu_{A})^{T}\sum_{A}^{-1}(x-\mu_{A})) +\frac{1}{3}\frac{1}{(2\pi)^{\frac{n}{2}} |\sum_{B}|^\frac{1}{2}} exp(-\frac{1}{2}(x-\mu_{B})^{T}\sum_{B}^{-1}(x-\mu_{B})) +\frac{1}{3}\frac{1}{(2\pi)^{\frac{n}{2}} |\sum_{C}|^\frac{1}{2}} exp(-\frac{1}{2}(x-\mu_{C})^{T}\sum_{C}^{-1}(x-\mu_{C}))]$

그림 2에서 training 결과를 볼 수 있다. Epoch이 지남에 따라 3개의 gaussian 분포로 나뉘는 것을 볼 수 있다. 

![Figure2](/assets/IWAE_img/Figure2.jpg) (그림 2 result)

## 2. Prior Sampling: Approximation for Large K

만일 training objective (식 4)의 K값이 너무나도 커서 위의 예제처럼 전부 다 summation을 진행하는 게 불가능하다면 어떻게 해야 할까?
아래의 식 5처럼 z값을 샘플링 (prior sampling)해서 Monte Carlo방식으로 approximate하는 방법 밖에는 없을 것이다. 그 이후에는 동일하게 gradient descent 방식으로 $\theta$를 학습하면 된다.

$\sum_{i}^{N}log \sum_{z}^{K}p_{z}(z)p_{\theta}(x^(i)|z) \approx \sum_{i}^{N}log \frac{1}{K}\sum_{k=1}^{K}p_{\theta}(x^(i)|z_{k}^{(i)})     z_{k}^{(i)} ~ p_{z}(z)$ (식 5)  

이것이 IWAE이전의 VAE모델들이 채택한 방식이다. 하지만 이와 같은 방식은 K값이 클수록 중요하지 않은 샘플들이 자주 추출되는 문제가 발생한다. 
**따라서 IWAE는 train objective에 $\frac{1}{K}\sum_{k=1}^{K}p_{\theta}(x^(i)|z_{k}^{(i)}$의 과정에서 중요하지 않은 샘플들이 뽑히는 문제를 다룬다**

예를들어 그림 3과 같이 데이터가 K개의 cluster들로 나뉘어 있고, $x^{(i)}$가 마지막 cluster에 있다고 가정한다. 그러면 prior distribution z에서 uniform sampling을 진행했을 때 $\frac{1}{K}$ term만이 유용할 것이다. 

![Figure3](/assets/IWAE_img/Figure3.jpg) (그림 3 K-Clusters)

이해를 돕기 위해 likelihood evaluation을 뜻하는 3번식 $p_{theta}(x) = \sum_{z} P_{z}(z)p_{\theta}(x|z)$을 다시 한 번 본다. 만일 우리가 MNIST데이터를 다루고 있고 (K=10), 숫자 1에 해당하는 데이터  $x^{(i)}$를 추출했다면, $p_{theta}(x) = \frac{1}{10}p_{theta}(x^{(i)}|z_{1}) + \frac{1}{10}\times0 + ... + + \frac{1}{10}\times0$가 될 것이다. $p_{theta}(x^{(i)}|z_{2})$를 포함한 다른 cluster에서 숫자 1에 해당하는 데이터가 뽑힐 확률은 0이기 때문이다.  

따라서 K값이 클수록 대부분의 샘플들은 meaningless해진다. 수천번의 샘플링을 하고도 training에 도움이 전혀 안된다면 문제가 있다고 할 수 있다.


## 3. Importance Sampling: How to Sample Meaningful Data
  
우리의 목적을 다시 한 번 상기하자면, $\E_{z~pz(z)}[p_{\theta}(x^{(i)}|z_{k}^{i})]$를 구하고 싶은 것이다. 그림 4를 통해 어떠한 경우에 문제가 되는 지 다시 한 번 보겠다.
4번과 5번 그림은 Mutual Information 채널의 Importance Sampling동영상에서 가져왔음을 밝힌다. 

![Figure4](/assets/IWAE_img/Figure4.jpg) (그림 4 Problem Case)

그림 4처럼 $p_{\theta}(x^{(i)}|z_{k}^{i})$가 높은 곳에서 $p_{z}(z)$가 낮으면 문제가 된다. $p_{z}(z)$에 따라 샘플된 데이터가 informative하지 않기 때문이다. 
$p_{\theta}(x^{(i)}|z_{k}^{i})$가 논문을 포함한 VAE모델에서 recognition network라고 지칭한다는 것을 알고 있다면, “VAE harshly penalizes approximate posterior samples which are unlikely to explain data, even if the recognition network puts much of its probability mass on good explanations.”라는 문장을 여기서 이해할 수 있다. VAE는 $p_{\theta}(x^{(i)}|z_{k}^{i})$값이 높더라도 $p_{z}(z)$값이 낮다면 해당 샘플을 사용하지 못한다.  

이러한 경우에 Importance Sampling기법을 사용해서 $p_{\theta}(x^{(i)}|z_{k}^{i})$에 대해 informative한 q distribution을 새롭게 정의하고, q distribution을 따라 샘플링한 데이터로 $p_{z}(z)$에 대한 기댓값을 찾는다. 
Importance sampling의 수식 유도 과정은 다음과 같다. 편의를 위해 discrete한 데이터 분포를 가정하고, $p_{\theta}(x^{(i)}|z_{k}^{i})$를 $f(z)$로 놓는다.

**Importance Sampling Formulation**

$\E_{z~pz(z)}[f(z)] 
      = \sum_{z}p_{z}(z)f(z)
      = \sum_{z}\frac{q(z)}{q(z)}p_{z}(z)f(z)
      = \sum_{z~q(z)}\frac{p_{z}(z)}{q(z)}f(z)
      = \E_{z~q(z)}[\frac{p_{z}(z)}{q(z)}f(z)
      \approx \frac{1}{K}\sum_{i=1}^{K}\frac{p_{z}(z^{(i)})}{q(zx^{(i)})f(z^{(i)}) with z^{(i)} ~ q(z) $ (식 6)
 
 식 6에서 볼 수 있듯이 이제는 $z^{i}$를 $q(z)$에서 샘플링하면서 원래의 train objective값을 구할 수 있다. 따라서 원래의 train objective가 4번식 $\sum_{i}^{N}log \sum_{z}^{K}p_{z}(z)p_{\theta}(x^(i)|z)$이었다면, 다음과 같이 변경된다. $\approx \sum_{i}log\frac{1}{K}\sum_{k=1}^{K}\frac{p_{z}(z_{k}^(i))}{q(z_{k}^{(i)})}p_{\theta}(x^{(i)}|z_{k}^{(i)}) with z_{k}^{(i)} ~ q(z_{k}^{(i)})$로 바뀌게 된다. 
 
우리는 이러한 과정을 $f(z)$가 높은 곳에서 높은 값을 가지는 $q(z)$를 찾는 문제로 보았지만, 조금 다른 관점으로 $f(z) \times q(z)$가 높도록 $q(z)$를 설정하는 문제로도, $Var_{z~q}[\frac{p(z)}{q(z)}f(z)] < Var_{z~p}[f(z)]$가 되도록 하는 문제로도 볼 수 있다. 여기서 $\frac{p}{q}$로 $f(z)$를 reweight해주는 걸로 볼 수 있는데, 아래의 그림 5로 그럴 때 $p(z)$와 같아지는 것을 볼 수 있다. 또한 초록색 q확률을 따라 reweighted된 주황색 값을 뽑으면 그 variance가 매우 작음을 그림을 통해 확인 할 수 있다. 

![Figure5](/assets/IWAE_img/Figure5.jpg) (그림 5 Solved Case)

## 4. Variational Approach for q(z): Ammortized Inference

이제 문제는 위에서 밝힌 조건들을 만족하는 좋은 q(z)를 찾는 것으로 바뀐다. 결국 샘플 $x^{(i)}$가 주어졌을 때 어떤 z가 informative한지 z를 uniform이 아닌 z sampler 를 통해 선정하는 것으로 볼 수 있다. 물론 Bayes rule을 사용하면 $p_{\theta}(z|x^{(i)}) = \frac{p_{\theta}(x^{(i)}|z) p_{z}(z)}{p_{\theta}(x^{(i)})}$가 되지만, 분모의 normalizing constant를 얻을 수 없기 때문에 $q(z)$를 샘플링하기 쉬운 known distribution으로 설정하는 variational approach를 사용한다. 논문에서는 Gaussian으로 설정하였고, KL divergence를 사용하여 근사시켰다. 전개하면 다음과 같다.

$min_{q(z)}KL(q(z)||p_{\theta}(z|x^{(i)}))
  = min_{q(z)}\E_{z~q(z)}log(\frac{q(z)}{p_{\theta}(z|x^{(i)})})
  = min_{q(z)}\E_{z~q(z)}log(frac{q(z)}{p_{\theta}(x^{(i)}|z)p_{z}(z)/p_{\theta}(x^{(i)})})
  = min_{q(z)}\E_{z~q(z)}[logq(z)-logp_{z}(z)-logp_{\theta}(x^{(i)}|z)]+logp_{\theta}(x^{(i)})$ 

이와같은 새로운 objective를 얻을 수 있다. 각 term을 모두 계산하는 것은 가능하지만 각 데이터 $x^{(i)}$마다 q를 찾는 건 비효율 적이기 때문에 같은 inference problem을 줄이기 위한 방법으로 Ammortized Inference방식을 사용해 효율적인 계산을 사용한다. 즉, q분포를 $\phi$ 파라미터를 가지는 Neural Network로 표현하는 것이다. 
결국 다음과 같은 $min_{\phi}\sum_{i}KL(q_{\phi}(z|x^{(i)})||p_{\theta}(z|x^{(i)}))$ (식 8)로 q를 찾아낸다. 

논문에서 “The recognition network generates multiple approximate posterior samples, and their weights are averaged.”는 이와 같은 f(z)에 informative한 amortized inferenced된 q분포에서 샘플을 얻는 부분을 의미한다. 


이제 q(z)는 reparameterization trick을 사용한 gaussian 분포로 $q_{\phi} = N(\mu_{\phi}(x), \sigma_{\phi}^{2}(x)), Equivalently: z = \mu_{\phi}(x)+\epsilon\sigma_{\phi}(x), \epsilon ~ N(0,I)$로 표현 될 수 있고, 그림 1의 Latent variable Model에 점선으로 표현된 inference 과정을 의미한다. 

## 5. Final Objective Function of IWAE

결론적으로 IWAE의 objective function은 latent variable model의 objective인 4번식 $max_{\theta}\sum_{i}^{N}logp_{\theta}(x^(i)) = \sum_{i}^{N}log \sum_{z}^{K}p_{z}(z)p_{\theta}(x^(i)|z)$에서 Importance Sampling을 통해 $\approx \sum_{i}log\frac{1}{K}\sum_{k=1}^{K}\frac{p_{z}(z_{k}^(i))}{q(z_{k}^{(i)})}p_{\theta}(x^{(i)}|z_{k}^{(i)}) with z_{k}^{(i)} ~ q(z_{k}^{(i)})$ 다음과 같이 변형된 것과, ammortized inference를 사용해 q분포를 제한다면서 8번식 $min_{\phi}\sum_{i}KL(q_{\phi}(z|x^{(i)})||p_{\theta}(z|x^{(i)}))$을 동시에 사용하게 된다. 

따라서 final objective function은 $max_{\theta,\phi}(\sum_{i}log\frac{1}{K}\sum_{k=1}^{K}\frac{p_{z}(z_{k}^(i))}{q(z_{k}^{(i)})}p_{\theta}(x^{(i)}|z_{k}^{(i)}) - \sum_{i}KL(q_{\phi}(z|x^{(i)})||p_{\theta}(z|x^{(i)})))$이 된다. 

## 6. Variational Auto-Encoder(VAE)와의 비교

IWAE는 기존의 VAE방식의 발전된 방법론으로 알려져 있다. 둘의 직접적인 비교를 위해 VAE를 짧게 소개한다. 추가적인 설명은 이번 포스팅 이전에 있는 VAE와 SBAI(Stochastic Backpropagation and Approximate Inference in DGM) 포스팅을 통해 학습할 것을 추천한다.



  
