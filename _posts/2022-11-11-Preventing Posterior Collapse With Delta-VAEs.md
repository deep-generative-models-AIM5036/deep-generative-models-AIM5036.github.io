---
layout: post
title:  "Preventing Posterior Collapse With Delta-VAEs"
date:   2022-11-11
author: Taemin Kim
categories: Latent-Variable
tags: VAE, Posterior Collapse, Delta-VAEs 
use_math: true
---

# **Preventing Posterior Collapse With Delta-VAEs**

## Ali Razavi, Aaron van den Oord, Ben Poole, Oriol Vinyals [ICLR 2019]
---
### Contents  
(Contents의 순서는 논문의 순서를 따름)  

0. Background
  - Posterior Collapse
1. Introduction
2. Mitigating Posterior Collapse with δ-VAEs
  - δ-VAE with Sequential Latent Variables
  - Anti-Causal Encoder Network
3. Experiments
  - Natural Images
  - Utilization of Latent Variables
  - Ablation Studies
4. Conclusions
5. Future Work
---
### 0. Background
 - Posterior Collapse란?  
  : Posterior Collapse는 Variational Auto Encoder(VAE)에서 주요한 문제 중 하나다. variational distribution이 prior distribution과 거의 유사해 지면서 decoder가 encoder에서 나온 latent variable에 대한 정보를 사용하지 못하게 되고, 따라서 VAE의 sample generate 능력을 감소시키게 되는 문제이다.  
> (James Lucas, George Tucker, Roger Grosse, Mohammad Norouzi, [“Don’t Blame the ELBO! A Linear VAE Perspective on Posterior Collapse”](https://proceedings.neurips.cc/paper/2019/file/7e3315fe390974fcf25e44a9445bd821-Paper.pdf) NeurIPS 2019.)
---
### 1. Introduction
 - Prior Work  
  : Autoregressive Model과 Representation Learning의 장단점은 아래와 같다
   * Autoregressive Model  
       장점 - 지역적인(local) 정보를 얻는데 용이하다.  
       단점 - 전체적인(global) 구조를 파악하는데 어려움이 있다.
   * Representation Learning  
       장접 - 전체적인(global) 구조를 파악이 용이하다.  
       단점 - 복잡하고 지역적인(local) 구조를 파악하는데 어려움이 있다.
   
   따라서 이 두가지 모델을 조합하여 장점을 극대화 하는 연구들이 많이 진행되었고, 좋은 결과(High quality)를 만들어내고 있다.
   하지만 강력한 Autoregressive Decoder를 사용하게 되면, generating 과정에서 Encoder에서 생성된 Latent Variable들이 무시되는 **Posterior Collapse** 현상이 발생하게 된다.
   일반적으로 Posterior Collapse를 해결하기 위해 일반적으로 아래 두 가지 방법이 많이 연구되었다.  
    1) Objective를 개선하는 방법
    2) 약한 Decoder를 사용하는 방법
  
 - Proposed Work  
  : 본 논문은 위에 언급한, Posterior Collapse를 해결하기 위한 일반적인 방법을 사용하지 않고(즉, 강력한 Decoder를 사용하고 Objective를 바꾸지 않으면서) Posterior Collapse를 개선하는 방법을 제안한다. 
---
### 2. Mitigating Posterior Collapse with δ-VAEs
 #### * δ-VAE 
   - VAE의 Training 과정은 ELBO를 최대화 하는 것이다. [Kingma & Welling, 2013](https://arxiv.org/pdf/1312.6114.pdf)  
     ELBO는 아래와 같이 표현된다.  

$$
\begin{align}
log\ p(x) = E_{q(z|x)}[log\ p(x, z) - log q(z|x)] + D_{KL}((q(z|x)|(p(z|x))) \\
\geq E_{q(z|x)}[log\ p(x, z) - log\ q(z|x)] \\
= E_{q(z|x)}[log\ p(x|z)] - D_{KL}(q(z|x)|p(z)) \\
\end{align}
$$

$$
\therefore ELBO = E_{q(z|x)}[log\ p(x|z)] - D_{KL}(q(z|x)|p(z)) 
$$
 
   - Posterior Collapse는 KL Divergence term에 의해서 발생한다고 알려져 있다. approximate posterior인 q(z|x)가 prior인 p(z)와 동일해 지면 KL Divergence term이 0으로 수렴하게 되고, generate할 때 Latent Variable은 input x에 대한 아무런 정보를 전달할 수 없게 되는 것이다.
   - δ-VAE는 KL Dievergence가 0이 되지 않도록 lower bound **δ**를 설정해 주는 것이다. 논문에서는 δ를 **committed rate**라고 부른다. 따라서 KL Divergece는 아래 식과 같은 제약조건을 가지게 된다. 

$$
min {\theta, \phi} D_{KL}(q_{\theta}(z|x) | p_{\theta}) \geq {\delta} 
$$

#### 2.1 δ-VAE with Sequential Latent Variables
이미지, 오디오, 텍스트와 같은 데이터들은 강한 공간-시간 연속성(strong spatio-temporal continuity)를 가지고 있다. 따라서 본 논문에서는 이러한 관계를 Latent Variable들이 표현해 줏 수 있도록 Sequential Latent Varivables 방법으로 Latent variable들 간의 관계를 설정하였다. Sequential Latent Variable을 표현하기 위해서 Prior의 latent variable들은 1차 선형 autoregressive(first-order linear qutoregressive progress)한 관계를 갖는다.(Z<sub>t</sub> = αZ<sub>t-1</sub> + ε<sub>s</sub>) 이 관계를 정리한 Prior는 다음과 같다.다음은 Sequential Latent Variable setting이다.
 
- Posterior
   
$$
q(z_{t}|x) = N(z_{t}; \mu_{t}(x), \sigma_{t}(x))
$$

- Prior  
 
$$
p(Z_{1}) = N(0, 1)
$$

$$
p(Z_{i}|Z_{i-1}) = N(αZ_{i-1}, \sqrt{1-\alpha^{2}}),\ if\ i > 1
$$

위의 Posterior와 Prior 관계의 mismatch를 두 분포에 대한 KL-Dievergence의 Lower Bound로 표시하면 아래 식과 같다. 여기서 mismatch가 δ로 표현된다.

$$
D_{KL}(q(z|x)||p(z)) \geq {1 \over 2} \sum_{k=1}^{d} (n-2)ln(1+\alpha_{k}^{2}) - ln(1-\alpha_{k}^{2}) = \delta
$$

여기서 n은 autoregressive time sequence의 길이이고, d는 latent variable의 dimension이다. **만약 d를 고정하면 δ는 n과 α로 표현할 수 있고** 그 관계를 아래 그래프와 같이 표현된다.

<그림>

오른쪽 그림은 α가 커지면서 prior의 Autoregressive한 성질이 점정 강해지고(즉, prior latent variable들 간의 correlation이 커진다.) 있는 것을 표현한 것이다. 그리고 correlation이 증가할 수록 δ도 증가하는 것을 확인할 수 있다.

#### 2.2 Anti-Causal Encoder Network


---
### 3. Experiments
---
### 4. Conclusions
---
### 5. Future Work
