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
 - δ-VAE 
   - VAE의 Training 과정은 ELBO를 최대화 하는 것이다. [Kingma & Welling, 2013](https://arxiv.org/pdf/1312.6114.pdf) ELBO는 아래와 같이 표현된다.  
   - log p(x) = Eq(z|x)[log p(x, z) − log q(z | x)] + DKL(q(z | x)||p(z | x))  
   &nasp;&nasp;&nasp;&nasp;&nasp;&nasp;≥ Eq(z|x)[log p(x, z) − log q(z | x)]   
   &nasp;&nasp;&nasp;&nasp;&nasp;&nasp;= Eq(z|x)[log p(x | z)] − DKL(q(z | x)||p(z))  
---
### 3. Experiments
---
### 4. Conclusions
---
### 5. Future Work
