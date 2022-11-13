---
layout: post
title:  "Residual Flows for Invertible Generative Modeling"
date:   2022-11-13
author: 이민영 & 이주엽
categories: Flow
use_math: true
---


# Residual Flows for Invertible Generative Modeling

이번 포스터에서는 "[Residual Flows for Invertible Generative Modeling](https://proceedings.neurips.cc/paper/2019/file/5d0d5594d24f0f955548f0fc0ff83d10-Paper.pdf)" 논문을 리뷰하겠습니다. 
본 논문은 Neurips 2019에 실린 논문으로 flow-based generative model 중 하나인 Residual Flow를 소개하고 있습니다. 

---

### 1. Introduction
Flow-based Generative model은 latent variable $z$를 역변환하여(invertible transformation) 데이터 $x$의 분포를 학습하는 생성모델 방법 중 하나입니다. 
역변환을 할 때 사용되는 개념이 change of variable로 뒤에서 한번 더 설명드리겠지만, 데이터 $x$의 확률을 
$log p(X) = log p(f(X)) + log |det\frac{\partial z}{\partial x}|$
의 형태로 계산합니다. 이때 필요한 조건은 $Z = f(X)$를 만족하는 $f(X)$가 invertible해야 한다는 것입니다.   

역변환을 계산하는 과정에서는 함수 f(X)에 대한 자코비안(Jacobian)을 계산하게 되는데, 기존 flow-based model에서는 계산을 쉽게 하기 위해 자코비안 행렬이 sparse하거나 삼각행렬과 같은 특수한 모양이 나오도록 하였습니다. 그러나, 자코비안이 sparse하거나 특정 모양을 따르게 되면 효율적으로 계산을 할 수는 있지만, 그런 조건을 만족시키는 함수 $f(X)$를 설계하는 것이 어렵고 비용이 많이 발생한다는 단점이 있습니다.   

또한 기존에 사용한 coupling block이나 ordinary differential equation 방법의 경우 강한 inductive bias를 야기하여 학습한 task이외의 task에는 적용하기 어렵다는 한계 역시 존재합니다.
그래서 이런 기존의 flow-based model이 가진 단점을 해결하고자 한 것이 <mark style='background-color: #fff5b1'> Residual Flow </mark> 모델입니다.   

---

### 2. Background
Residual Flow 모델을 알아보기에 앞서 이해하는데 도움이 될 몇가지 개념들을 간단히 소개하도록 하겠습니다.   

##### 2.1. Change of Variable
random variable X와 Z가 $X \~ p(x)$,  $Z \~ p(z)$ 의 분포를 따른다고 할때
$X = f(Z)$이면 $Z = f^{-1}(X)$가 되어 다음과 같이 변환할 수 있습니다.

$$
\begin{align}
    p(x) &= p(z)|\frac{dz}{dx}| \\
    &= p(f(x))det|\frac{df(x)}{dx}|
\end{align}
$$

이를 이용하여 데이터에 대한 log density인 $log p(x)$는 다음과 같이 표현됩니다.

$$
log p(x) = log p(f(x)) + log |det\frac{df(x)}{dx}|
$$

이러한 change of variable은 flow-based model의 핵심으로 이때 $f(x)$는 invertible한 함수여야 합니다.   

change of variable을 통해 flow-based generative model은 ELBO를 통해 간접적으로 $log p(x)$값을 최대화하는 VAE(Variational AutoEncoder)나 데이터의 분포 추정없이 실제 데이터와 생성된 데이터를 분류하는 discriminator를 통해 생성 모델을 학습하는 GAN(Generative Adversarial Network)와는 다르게 데이터의 log density를 직접 추정할 수 있다는 장점이 있습니다.



##### 2.2 Lipschitz constraint   

립시츠 조건은 어떤 함수 내 임의의 두 점사이의 거리가 일정 비 이상이 되지 않도록 하는 조건으로 다음 식과 같이 나타낼 수 있고, 식에서 두 점 사이의 거리의 비를 제한한 상수 k를 립시츠 상수(Lipschitz constant)라고 합니다.

$$
\frac{|f(x_1)-f(x_2)|}{|x_1 - x_2|} \leq k
$$


##### 2.3 Log det(J) = tr(log J)   

determinant의 log값을 trace로 변환할 수 있는 방법입니다. 

$$
    A = U \cdot D \cdot U^{-1}  \qquad   f(A) = U\cdot f(D) \cdot U^{-1} \\
$$

$$
    \begin{aligned}
        det(f(A)) &= det(U) \cdot det(f(D)) \cdot \frac{1}{det(U)} \\
        &= det(f(D)) \\
        &= \prod_af(\lambda_a) \\
    \end{aligned}
$$


$$
    \begin{aligned}
        tr(f(A)) &= tr(U \cdot f(D) \cdot U^{-1} \\
        &= tr(f(D)) \\
        &= \sum_af(\lambda_a) \\
    \end{aligned}
$$

$$
    \begin{aligned}
        &=> det (exp (A)) = \prod_aexp(\lambda_a) = exp \sum_a\lambda_a = exp (tr(A)) \\
        &=> det(J) = exp (tr(logJ)) \qquad \qquad  A = log J \\
        &=> log (det(J)) = tr(log J)
    \end{aligned}
$$


##### 2.4 Newton-Mercator series

뉴턴 메르카토르 급수는 자연로그에 대한 테일러 급수로 $x$의 값이 $-1 < x \leq 1$ 인 경우에 다음과 같이 성립하는 급수입니다.

$$
    \begin{aligned}
        log (1 + x) &= x - \frac{x^2}{2} + \frac{x^3}{3} - \frac{x^4}{4} + \cdots \\
        &= \sum_{n=1}^{\infty}\frac{(-1)^{n+1}}{x}x^n \qquad \qquad \qquad  for -1< x \leq 1
    \end{aligned}
$$

##### 2.5 Russian Roullete Estimator

러시안 룰렛 추정히는 추정 값 $\hat{Y}$을 계산하는 과정에서 확률 $p$에 의해 계산을 중단할지 결정한 뒤 계산이 중단되면 그때까지 추정된 값을 Y의 추정치로 반환해주는 것을 말합니다. 이를 식으로 나타내면 다음과 같습니다.

$$
    \begin{aligned}
        \hat{Y} = \hat{Y} + \frac{\Delta_t}{(1-p)^t} \\
    \end{aligned}
$$

$p$ : probability to halt  estimate

##### 2.6 Skillin-Hutchinson Estimator

스킬링-허친슨 추정은 행렬의 대각합(trace)를 추정하는 방법입니다. 가우시안 분포에서 뽑은 벡터 z를 이용하여 다음과 같이 행렬의 대각합을 추정할 수 있습니다.

$$
    \begin{aligned}
        tr(A) &= tr(A \cdot I) \\
        &= tr(A \cdot E(zz^T)) \\
        &= E[tr(Azz^T)] \\
        &= E[z^TAz] \\ 
        z \thicksim Gaussian
    \end{aligned}
$$

##### 2.7 Neumann seires

노이만 급수는 행렬의 역수에 대한 식으로 다음과 같이 표현할 수 있습니다.

$$
    (I-A)^{-1} = \sum_{n = 0}^\infty A^n
$$

##### 2.8 Invertible Residual Network (i-ResNet)

i-ResNet[^1]은 image classification에서 사용되는 Residual Netowrk를 invertible하게 만들 수 있는 방법을 제안한 모델입니다.
Residual Network의 경우 $f(x) = x + g(x)$의 형태로 네트워크가 구성되어 있는데, 
$g(x)$ 함수에 임의의 두 점에 대해 그 거리의 비가 1보다 작거나 같아야 한다는 unity Lipschitz constraint를 주어 inverible하게 만들었습니다. 

$$
    \begin{aligned}
        log(p(x)) &= log(p(f(x)) + log(|det(J_F(x))|) \\
        &= log(p(f(x))) + tr(log(J_F(x)))    \because (2.3 Log det(J) = tr(log J))   \\
        &= log(p(f(x))) + tr(log(I + J_g(x))))  \because (ResNet) \\
        &= log(p(f(x))) + tr(\sum_{k=1}^\infty \frac{(-1)^{k+1}}{k}[J_g(x)]^k) \because (2.4 Newton-Mercator series) \\
        Lip(g) \leq 1
    \end{aligned}
$$

위와 같은 과정을 통해 $log(p(x)) = log(p(f(x))) + tr(\sum_{k=1} (\frac{(-1)^{k+1}}{k}[J_g(x)]^k))$로 나타냄으로써 자코비안이 sparse하거나 structured form이 아닌 함수를 사용하여도 change of variable을 적용가능한 flow를 보여준 모델입니다. 

---

### 3. Residual Flow

##### 3.1 Unbiased Log Density Esitmation

2.8절에서 설명한 i-ResNet의 식은 다음과 같습니다.

$$
log(p(x)) = log(p(f(x))) + tr(\boxed{\sum_{k=1}^\infty (\frac{(-1)^{k+1}}{k}[J_g(x)]^k)})
$$

식에서 박스가 쳐진 부분은 무한급수의 형태로 계산하기가 어렵다는 한계가 있었습니다.
그래서 이를 해결하고자 기존의 방법들은 $n$번째까지만 계산하여 무한 급수의 값을 추정하는 방법을 사용하였습니다.
그러나 이때 $n+1$번째이후로의 계산되는 부분이 bias값이 되어 biased estimator를 사용할 수 밖에 없었습니다.
아래 식에서 우항의 앞 부분이 급수를 추정하기 위해 사용되는 1부터 n번째까지의 합이고, 우항의 뒷 부분은 계산을 생략한 n+1번째부터의 합을 의미하는데, 여기서 추정값의 표현력을 높이기 위해서는 n의 값을 키워 더 많은 값을 계산할 수 있지만 n이 커질수록 계산량이 늘어나게 됩니다.
반대로 계산량을 줄이기 위해서 n의 값을 줄인다면 계산을 하지 않는 부분이 많아져 bias가 높아지게 되어 expressive와 bias사이의 trade-off가 발생하게 되는 것입니다.

$$ 
\sum_{k=1}^\infty (\frac{(-1)^{k+1}}{k}[J_g(x)]^k)  = {\sum_{k=1}^n (\frac{(-1)^{k+1}}{k}[J_g(x)]^k)} + {\sum_{k=n+1}^\infty (\frac{(-1)^{k+1}}{k}[J_g(x)]^k)}
$$

그래서 Residual Flow는 계산을 진행한 n을 미리 정하는 것이 아니라, 다음 계산을 계속 할지 그만둘지를 확률에 기반하여 선택하는 방법을 사용하였습니다. 
$\Delta_k = \frac{(-1)^{k+1}}{k}[J_g(x)]^k$로 나타내고, $b \thicksim Bernoulli(q)$를 이용하여 계산을 그만둘지 여부를 결정하게 됩니다. 

이를 식으로 표현하면 다음과 같은데, 여전히 bias가 존재하는 문제가 있습니다.

$$
\Delta_1 + \mathbb{E}[(\sum_{k=1}^\infty\Delta_k \mathbb{1}_{b=0} + (0) \mathbb{1}_{b=1}] = \Delta_1 + \sum_{k=2}^\infty \Delta_k(1-q) \neq \sum_{k=1}^\infty\Delta_k
$$

그래서 이를 unbiased estimator로 만들기 위해서 $1-q$로 나누어 아래 식과 같이 unbiased estimator를 사용할 수 있습니다.

$$
\Delta_1 + \mathbb{E}[\frac{\displaystyle\sum_{k=2}^\infty \Delta_k}{\boxed{1-q}} \mathbb{1}_{b=0} +(0)\mathbb{1}_{b=1}] = \Delta_1 + \frac{\displaystyle\sum_{k=2}^\infty \Delta_k}{\boxed{1-q}}(1-q) = \sum_{k=1}^\infty\Delta_k 
$$

위 식을 참조하면 매번 q의 확률로 계산을 멈출 지 결정하게 되고, $1-q$의 확률로 계산을 하기로 결정이 된다면 아래의 식과 같이 나태낼 수 있습니다.

$$
\begin{aligned}
    &\Delta_1 + \mathbb{E}[\frac{\overset{\infty}{\underset{k=2}{\sum}} \Delta_k}{1-q}1_{b=0}  +(0)1_{b=1}]   \\
    &\Rightarrow \Delta_1 + \frac{\Delta_2}{1-q} \mathbb{E}[\frac{\overset{\infty}{\underset{k=3}{\sum}} \Delta_k}{1-q}1_{b=0}  +(0)1_{b=1}]   \\ 
    &\Rightarrow \Delta_1 + \frac{\Delta_2}{1-q} + \frac{\Delta_3}{(1-q)^2} \mathbb{E}[\frac{\overset{\infty}{\underset{k=4}{\sum}} \Delta_k}{1-q}1_{b=0}  +(0)1_{b=1}]   \\
    &\Rightarrow \frac{\Delta_1}{\mathbb{P}(N\geq 1)} + \frac{\Delta_2}{\mathbb{P}(N\geq 2)} +\frac{\Delta_3}{\mathbb{P}(N\geq 3)} + \cdots \frac{\Delta_n}{\mathbb{P}(N\geq n)} = \sum_{k=1}^{n}\frac{\Delta_k}{\mathbb{P}(N\geq k)}
\end{aligned}
$$

이를 이용하여 $log p(x)$를 나타내면

$$
\begin{aligned}
log(p(x)) &= log(p(f(x))) + tr(\sum_{k=1}^\infty (\frac{(-1)^{k+1}}{k}[J_g(x)]^k)) \\
&= log(p(f(x))) + tr(\sum_{k=1}^{n}\frac{(-1)^{k+1}}{k}\frac{[J_g(x)]^k}{\mathbb{P}(N\geq k)})
\end{aligned}
$$

이와 같이 추정할 수 있고, 여기에서 Skilling-Huchinson Trace Estimator를 적용하면 최종적으로 아래의 식과 같이 $log p(x)$를 추정할 수 있습니다.

$$
\begin{aligned}
&log(p(x)) = log(p(f(x))) + \mathbb{E}[\sum_{k=1}^{n}\frac{(-1)^{k+1}}{k}\frac{v^T[J_g(x)]^kv}{\mathbb{P}(N\geq k)}] \\
&f(x) = x + g(x)  \text{ with }   Lip(g) < 1 \\
&n \thicksim P(N), v \thicksim \mathcal{N}(0, I)
\end{aligned}
$$


이 때 립시츠 조건에 의해서 $J_g(x)^k$ 가 우하향 exponential 함수의 그래프를 따르게 되어 빠른 수렴이 가능합니다. 계산을 진행할 횟수인 $n$에 따라서 $log p(x)$의 추정값이 달라질 수 있기 때문에 $n$을 샘플링하는 $P(N)$에 따라 추정값의 variance가 높아질 수 있는데,  $J_g(x)^k$ 가 빠르게 수렴하면 $n$에 따른 variance 값 역시 무시할 수 있는 수준이 됩니다. 따라서 추정값의 variance를 줄이기 위해 $P(N)$을 tuning할 필요가 없게 되고, 본 논문에서는 $Geom(0.5)$로 고정하여 사용하였습니다.

그 결과가 아래에 있는 그림을 통해 확인할 수 있는데, 기존 i-ResNet에서 사용한 방식과 같이 계산할 횟수 n을 일정하게 정한 뒤 절삭하여 계산한 값으로 $log p(x)$를 추정한 결과가 빨간색 그래프이고, residual flow가 제안한 방법으로 $log p(x)$를 추정한 결과가 파란색 그래프로 나타나 있습니다. 그림을 보면 추정된 값 자체의 bits/dim은 기존의 방법인 빨간색이 더 적은 수치를 기록하여 더 좋은 결과를 보인다고 생각할 수도 있지만, 실제 $log p(x)$값을 나타내는 실선과 비교하였을 때 빨간색 그래프는 추정값과 실제값이 서로 맞지 않는, 즉 biased estimator인 것을 확인할 수 있는 반면, 파란색 그래프를 보면 실제 $log p(x)$의 값과 추정된 값이 서로 일치하는 것을 보았을 때 unbiased estimator로 추정을 하였음을 확인할 수 있습니다.

<img src="https://user-images.githubusercontent.com/76925973/200765508-1f58a9ee-e744-460d-92cc-38b1e77b283c.png"  width="400" height="250">


##### 3.2 Memory-Efficient Backpropagation

$$
log(p(x)) = log(p(f(x))) + \mathbb{E}[\boxed{{\sum_{k=1}^{n}}}\frac{(-1)^{k+1}}{k}\frac{v^T[\boxed{{J_g(x)}}]^kv}{\mathbb{P}(N\geq k)}]  \cdots (3.2.1)
$$

이와 같이 $log p(x)$를 추정한 unbaised estimator를 이용하여 모델을 학습시킬 때 backpropagation과정에서 메모리를 효율적으로 관리하는 것 역시 중요합니다. 위 식에서 첫번째 박스에서 n번의 계산을 해야하고, 두번째 박스에서 m개의 residual block을 계산해야하기 때문에 위의 식을 그대로 backpropagation에 이용하며 $O(n\cdot m)$ 메모리가 필요하게 됩니다.

따라서 본 논문에서는 메모리를 효율적으로 사용하기 위해서 $log p(x)$의 추정값을 그대로 backpropagation에서 사용하는 것이 아니라 unbiased log-determinatnt gradient estimator를 이용하였습니다. 

$$
\begin{aligned}
    &\frac{\partial}{\partial \theta_i} log det(I+J_g(x, \theta) \\
    &= \frac{1}{det(I+J_g(x, \theta)}[\frac{\partial}{\partial \theta_i}det(I+J_g(x, \theta)]  \text{ }\cdots (a) \\
    &= \frac{1}{det(I+J_g(x, \theta)}[det(I+J_g(x, \theta)tr\bigg(\big(I+J_g(x, \theta)\big)^{-1} \frac{\partial \big(I+J_g(x, \theta)\big)}{\partial \theta_i} \bigg)] \text{ }\cdots (b) \\
    &= tr\bigg(\big(I + J_g(x, \theta) \big)^{-1} \frac{\partial \big(I+J_g(x, \theta)\big)}{\partial \theta_i} \bigg) \\
    &= tr\bigg(\big(I + J_g(x, \theta) \big)^{-1} \frac{\partial \big(J_g(x, \theta)\big)}{\partial \theta_i} \bigg) \\
    &= tr\bigg(\bigg[\sum_{k=0}^\infty (-1)^k J_g(x, \theta)^k \bigg] \frac{\partial \big(J_g(x, \theta)\big)}{\partial \theta_i} \bigg)  \text{ }\cdots (c) \\
    &= tr\bigg(\sum_{k=0}^n \frac{(-1)^k}{P(n \geq k)}J_g(x, \theta)^k \frac{\partial \big(J_g(x, \theta)\big)}{\partial \theta_i} \bigg) \text{ }\cdots (d)  \\
    &= \mathbb{E}\bigg[ \Big( \sum_{k=0}^n \frac{(-1)^k}{P(N \geq k)}v^T J_g(x, \theta)^k \Big) \frac{\partial \big(J_g(x, \theta)\big)}{\partial \theta_i} v \bigg] \text{ }\cdots (e) \\
    & n \thicksim P(N), v \thicksim \mathcal{N}(0, I)
\end{aligned}
$$

위 식은 log determinant를 미분하는 과정을 보여주는 식으로 (a)는 log값을 미분하는 chain rule을 적용한 것이고, (b)는 determinant 미분을 적용한 식 입니다. (c)는 $(I-f)^{-1} = \displaystyle\sum_{k=0}^{\infty}f^k $ 를 만족한다는 Neumann series를 적용한 것이며, (d)는 Russian Roullete estimator를 (e)는 Skilling-Hutchinson trace estimator를 적용한 결과를 나타낸 것입니다.

이를 통해 최종적으로 log determinant 값을

$$
\frac{\partial}{\partial \theta_i} log det(I+J_g(x, \theta) = \mathbb{E}\bigg[ \Big( \sum_{k=0}^n \frac{(-1)^k}{P(N \geq k)}v^T J_g(x, \theta)^k \Big) \frac{\partial \big(J_g(x, \theta)\big)}{\partial \theta_i} v \bigg] \text{ }\cdots (3.2.2)
$$

로 나타낼 수 있고, 여기서는 (3.2.1)의 두번째 박스에서 계산해야 했던 m개의 residual block에 대한 power series 계산이 사라지기 때문에 메모리를 $O(n)$으로 줄일 수 있습니다.


본 논문에서는 메모리를 줄일 수 있는 방법을 한가지 더 제시하여 backward과정의 계산을 forward 과정에서 계산한 것을 사용한다는 것으로 Backward-in-Forward 를 제안하였습니다. 

$$
\frac{\partial L}{\partial \theta} = \underset{scalar}{\underbrace{\frac{\partial L}{\partial log det(I+J_g(x, \theta)}}}
\underset{vector}{\underbrace{\frac{\partial log det(I+J_g(x, \theta)} {\partial \theta}}} \cdots (3.2.3)
$$

Loss를 미분할때 log determinant의 미분을 사용한다면 위 (3.2.3) 식과 같이 Loss를 log determinant 값으로 미분하는 scalar부분과 log determinant를 미분하는 vector 부분으로 분리할 수 있는데, 이때 log determinant는 forward에서 계산을 하는 과정이므로 backward에서 다시 계산할 필요가 없다는 점에서 메모리를 줄일 수 있다는 것입니다.

<img src="https://user-images.githubusercontent.com/76925973/200817001-b285bac3-5a93-4697-a090-67a6c2408460.png"  width="400" height="250">

위 그림을 보면 파란색은 backpropagation을 그대로 진행한 경우를 나타내고 초록색은 unbiased log determinant gradient estimator (3.2.2)식을 이용한 경우, 빨간색은 backward-in-forward (3.2.3)을 이용한 경우이며, 보라색은 두가지 방법을 모두 사용한 경우의 메모리 사용량을 나타냅니다. 이를 통해 두가지 방법을 모두 사용하여 backpropagation을 계산하는 경우 메모리를 훨씬 효율적으로 사용할 수 있음을 확인할 수 있었고 본 논문에서도 역시 두가지 모두를 사용하여 메모리를 효율적으로 사용하고자 하였습니다.


##### 3.3 LipSwish Activation Function
Backpropagation을 진행할 때 메모리 뿐만 아니라 activation derivative saturation 문제 역시 발생할 수 있습니다. Jacobian을 계산하는 과정에서 일차 미분을 진행하게 되고 gradient를 계산하는 과정에서 이차 미분이 진행되는데, 이때 일차 미분값이 상수가 되면 이차 미분 값이 0이 되면서 gradient vanishing 문제가 발생하는 것입니다. 그래서 논문에서는 립시츠 조건을 만족시키면서 gradient vanishing 문제가 발생하지 않도록 하기 위한 activation function의 두가지 조건을 제시하였습니다.  

1) 일차 미분 값이 $|\phi'(z)| \leq 1 \qquad \forall z$   를 만족해야 한다.
2) $|\phi'(z)|$ 가 1과 가까운 점에서의 이차 미분 값이 0이 되어 vanish되서는 안된다.

대부분의 activation function이 첫번째 조건은 만족시키지만 두번째 조건을 만족시키기 어려워 본 논문에서는 두번째 조건을 만족시킬 수 있는 Swish function을 사용했다고 합니다. Swish activaton function은 $f(x) = x \cdot \sigma(\beta x)$ 로 밑의 그림 중 파란색으로 그려진 함수입니다. 그림을 통해서 알 수 있듯이 ReLU의 경우 미분값이 일정해지는 구간이 발생하여 이차 미분 시 gradient vanishing 문제를 야기하지만 Swish 함수의 경우 그렇지 않아 두번째 조건을 만족할 수 있습니다.

<img src="https://user-images.githubusercontent.com/76925973/200821177-8131b782-b746-445a-8b78-3295eba52e03.png"  width="400" height="200">

그러나 Swish 함수의 경우 일차 미분값이 $ | \frac{d}{dz} Swish(z) | \lesssim 1.1 $ 으로 첫번째 조건을 만족하지 않습니다. 그래서 본 논문은 Swish 함수를 1.1로 나누어 주어 첫번째 조건 역시 만족할 수 있는 LipSwish 함수를 만들었습니다.

$$
LipSwish(z) = \frac{Swish(z)}{1.1} = \frac{z \cdot \sigma(\beta z)} {1.1}
$$

이를 activation function으로 사용하였습니다. 이때 $\beta$ 값은 softplus를 통해 양수를 유지하도록 학습시켰습니다.

<img src="https://user-images.githubusercontent.com/76925973/200821897-be1bf86c-4666-4cd6-9ade-2001592f339f.png"  width="500" height="120">



