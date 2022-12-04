---
layout: post
title:  "Residual Flows for Invertible Generative Modeling"
date:   2022-11-13
author: 이민영, 이주엽
categories: ["Flow Models"]
tags: ["Flow Models"]
use_math: true
---



# Residual Flows for Invertible Generative Modeling

이번 포스터에서는 "[Residual Flows for Invertible Generative Modeling](https://proceedings.neurips.cc/paper/2019/file/5d0d5594d24f0f955548f0fc0ff83d10-Paper.pdf)" 논문을 리뷰하겠습니다. 
본 논문은 Neurips 2019에 실린 논문으로 flow-based generative model 중 하나인 Residual Flow를 소개하고 있습니다. 

---

### 1. Introduction
Flow-based Generative model은 latent variable $z$를 역변환하여(invertible transformation) 데이터 $x$의 분포를 학습하는 생성모델 방법 중 하나입니다. 
역변환을 할 때 사용되는 개념이 change of variable로 뒤에서 한번 더 설명드리겠지만, 데이터 $x$의 확률을 
$log p(X) = log p(f(X)) + log \vert det\frac{\partial z}{\partial x}\vert $
의 형태로 계산합니다. 이때 필요한 조건은 $Z = f(X)$를 만족하는 $f(X)$가 invertible해야 한다는 것입니다.   

역변환을 계산하는 과정에서는 함수 f(X)에 대한 자코비안(Jacobian)을 계산하게 되는데, 기존 flow-based model에서는 계산을 쉽게 하기 위해 아래 그림과 같이 자코비안 행렬이 sparse하거나 삼각행렬과 같은 특수한 모양이 나오도록 하였습니다. 그러나, 자코비안이 sparse하거나 특정 모양을 따르게 되면 효율적으로 계산을 할 수는 있지만, 그런 조건을 만족시키는 함수 $f(X)$를 설계하는 것이 어렵고 비용이 많이 발생한다는 단점이 있습니다.   


<img src="https://user-images.githubusercontent.com/117256746/205273638-c0d5c36e-e367-40b8-88a6-8acc34177f15.png"  width="400" >

[출처] Residual Flow [^1]

또한 기존에 사용한 coupling block이나 ordinary differential equation 방법의 경우 강한 inductive bias를 야기하여 학습한 task이외의 task에는 적용하기 어렵다는 한계 역시 존재합니다.
그래서 이런 기존의 flow-based model이 가진 단점을 해결하고자 한 것이 Residual Flow 모델입니다.   

---

### 2. Background
Residual Flow 모델을 알아보기에 앞서 이해하는데 도움이 될 몇가지 개념들을 간단히 소개하도록 하겠습니다.   

##### 2.1. Change of Variable
random variable X와 Z가 $X \thicksim p(x)$,  $Z \thicksim p(z)$ 의 분포를 따른다고 할때
$X = f(Z)$이면 $Z = f^{-1}(X)$가 되어 다음과 같이 변환할 수 있습니다.

$$
\begin{align}
    p(x) &= p(z)\vert \frac{dz}{dx}\vert  \\
    &= p(f(x))det\vert \frac{df(x)}{dx}\vert 
\end{align}
$$

이를 이용하여 데이터에 대한 log density인 $log p(x)$는 다음과 같이 표현됩니다.

$$
log p(x) = log p(f(x)) + log \vert det\frac{df(x)}{dx}\vert 
$$

이러한 change of variable은 flow-based model의 핵심으로 이때 $f(x)$는 invertible한 함수여야 합니다.   

change of variable을 통해 flow-based generative model은 ELBO를 통해 간접적으로 $log p(x)$값을 최대화하는 VAE(Variational AutoEncoder)나 데이터의 분포 추정없이 실제 데이터와 생성된 데이터를 분류하는 discriminator를 통해 생성 모델을 학습하는 GAN(Generative Adversarial Network)와는 다르게 데이터의 log density를 직접 추정할 수 있다는 장점이 있습니다.



##### 2.2 Lipschitz constraint   

립시츠 조건은 어떤 함수 내 임의의 두 점사이의 거리가 일정 비 이상이 되지 않도록 하는 조건으로 다음 식과 같이 나타낼 수 있고, 식에서 두 점 사이의 거리의 비를 제한한 상수 k를 립시츠 상수(Lipschitz constant)라고 합니다.

$$
\frac{\vert f(x_1)-f(x_2)\vert }{\vert x_1 - x_2\vert } \leq k
$$


##### 2.3 Log det(J) = tr(log J)   

determinant의 log값을 trace로 변환할 수 있는 방법입니다. 

$$
    A = U \cdot D \cdot U^{-1}  \qquad    f(A) = U\cdot f(D) \cdot U^{-1} \\
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
        &=> det(J) = exp (tr(logJ)) \qquad \qquad A = log J \\
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

i-ResNet[^2]은 image classification에서 사용되는 Residual Netowrk를 invertible하게 만들 수 있는 방법을 제안한 모델입니다.
Residual Network의 경우 $f(x) = x + g(x)$의 형태로 네트워크가 구성되어 있는데, 
$g(x)$ 함수에 임의의 두 점에 대해 그 거리의 비가 1보다 작거나 같아야 한다는 unity Lipschitz constraint를 주어 invertible하게 만들었습니다. 

바나흐 고정점 정리에 의하면 두 점 사이의 거리가 1보다 작다는 조건을 만족하는 함수는 유일한 고정점 한 개 갖게 됩니다. 이는 역함수 정리에서 함수가 열린 집합임을 증명하기 위해 필요한 조건으로 바나흐 고정점 정리가 만족한다면, 역함수 조건에 의해 단사함수와 열맂 집합 조건을 만족하게 되어 일대일 대응을 만족하는 함수가 되기 때문에 립시츠 조건을 만족하게 되면 invertible한 함수를 만들 수 있게 됩니다. [바나흐 고정점 정리](https://kty890309.tistory.com/15)와 [역함수 정리](https://ko.wikipedia.org/wiki/%EC%97%AD%ED%95%A8%EC%88%98_%EC%A0%95%EB%A6%AC)는 링크를 통해 자세한 내용을 참고하시기 바랍니다. 

$$
    \begin{aligned}
        log(p(x)) &= log(p(f(x)) + log(\vert det(J_F(x))\vert ) \\
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
아래 식에서 파란색 부분이 급수를 추정하기 위해 사용되는 1부터 n번째까지의 합이고, 초록색 부분은 계산을 생략한 n+1번째부터의 합을 의미하는데, 여기서 추정값의 표현력을 높이기 위해서는 n의 값을 키워 더 많은 값을 계산할 수 있지만 n이 커질수록 계산량이 늘어나게 됩니다.
반대로 계산량을 줄이기 위해서 n의 값을 줄인다면 계산을 하지 않는 부분이 많아져 bias가 높아지게 되어 expressive와 bias사이의 trade-off가 발생하게 되는 것입니다.

$$ 
\sum_{k=1}^\infty (\frac{(-1)^{k+1}}{k}[J_g(x)]^k)  = \color{blue}{\sum_{k=1}^n (\frac{(-1)^{k+1}}{k}[J_g(x)]^k)} + \color{green}{\sum_{k=n+1}^\infty (\frac{(-1)^{k+1}}{k}[J_g(x)]^k)}
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


이 때 립시츠 조건에 의해서 $J_g(x)^k$ 가 우하향 exponential 함수의 그래프를 따르게 되어 빠른 수렴이 가능합니다. 계산을 진행할 횟수인 $n$에 따라서 $log p(x)$의 추정값이 달라질 수 있기 때문에 $n$을 샘플링하는 $P(N)$에 따라 추정값의 variance가 높아질 수 있는데,  $J_g(x)^k$ 가 빠르게 수렴하면 $n$에 따른 variance 값 역시 무시할 수 있는 수준이 됩니다. 따라서 추정값의 variance를 줄이기 위해 $P(N)$을 tuning할 필요가 없게 되고, 본 논문에서는 $Geom(0.5)$로 고정하여 사용하였습니다. 이를 통해 아래 코드와 같이 샘플링을 통해 계산한 n을 구하고 geometric 분포의 확률을 계산한 값을 사용하여 power series estimator를 계산할 수 있습니다.

```python
def _logdetgrad(self, x):
    """Returns g(x) and logdet|d(x+g(x))/dx|."""

    with torch.enable_grad():
        if (self.brute_force or not self.training) and (x.ndimension() == 2 and x.shape[1] == 2):
            ###########################################
            # Brute-force compute Jacobian determinant.
            ###########################################
            x = x.requires_grad_(True)
            g = self.nnet(x)
            # Brute-force logdet only available for 2D.
            jac = batch_jacobian(g, x)
            batch_dets = (jac[:, 0, 0] + 1) * (jac[:, 1, 1] + 1) - jac[:, 0, 1] * jac[:, 1, 0]
            return g, torch.log(torch.abs(batch_dets)).view(-1, 1)

        geom_p = torch.sigmoid(self.geom_p).item()
        sample_fn = lambda m: geometric_sample(geom_p, m)
        rcdf_fn = lambda k, offset: geometric_1mcdf(geom_p, k, offset)

        if self.training:
            # Unbiased estimation.
            lamb = self.lamb.item()
            n_samples = sample_fn(self.n_samples)
            n_power_series = max(n_samples) + self.n_exact_terms
            coeff_fn = lambda k: 1 / rcdf_fn(k, self.n_exact_terms) * sum(n_samples >= k - self.n_exact_terms) / len(n_samples)


        ####################################
        # Power series with trace estimator.
        ####################################
        vareps = torch.randn_like(x)

        # neumann estimator.
        estimator_fn = neumann_logdet_estimator

        # Do backprop-in-forward to save memory.
        g, logdetgrad = mem_eff_wrapper(estimator_fn, self.nnet, x, n_power_series, vareps, coeff_fn, self.training)

        return g, logdetgrad.view(-1, 1)


def batch_jacobian(g, x):
    jac = []
    for d in range(g.shape[1]):
        jac.append(torch.autograd.grad(torch.sum(g[:, d]), x, create_graph=True)[0].view(x.shape[0], 1, x.shape[1]))
    return torch.cat(jac, 1)


def batch_trace(M):
    return M.view(M.shape[0], -1)[:, ::M.shape[1] + 1].sum(1)
    
######################## 
Geometric distribution 
########################    
def geometric_sample(p, n_samples):
    return np.random.geometric(p, n_samples)

def geometric_1mcdf(p, k, offset):
    if k <= offset:
        return 1.
    else:
        k = k - offset
    """P(n >= k)"""
    return (1 - p)**max(k - 1, 0)

# neumann estimator
def neumann_logdet_estimator(g, x, n_power_series, vareps, coeff_fn, training):
    vjp = vareps
    neumann_vjp = vareps
    with torch.no_grad():
        for k in range(1, n_power_series + 1):
            vjp = torch.autograd.grad(g, x, vjp, retain_graph=True)[0]
            neumann_vjp = neumann_vjp + (-1)**k * coeff_fn(k) * vjp
    vjp_jac = torch.autograd.grad(g, x, neumann_vjp, create_graph=training)[0]
    logdetgrad = torch.sum(vjp_jac.view(x.shape[0], -1) * vareps.view(x.shape[0], -1), 1)
    return logdetgrad
```

그 결과가 아래에 있는 그림을 통해 확인할 수 있는데, 기존 i-ResNet에서 사용한 방식과 같이 계산할 횟수 n을 일정하게 정한 뒤 절삭하여 계산한 값으로 $log p(x)$를 추정한 결과가 빨간색 그래프이고, residual flow가 제안한 방법으로 $log p(x)$를 추정한 결과가 파란색 그래프로 나타나 있습니다. 그림을 보면 추정된 값 자체의 bits/dim은 기존의 방법인 빨간색이 더 적은 수치를 기록하여 더 좋은 결과를 보인다고 생각할 수도 있지만, 실제 $log p(x)$값을 나타내는 실선과 비교하였을 때 빨간색 그래프는 추정값과 실제값이 서로 맞지 않는, 즉 biased estimator인 것을 확인할 수 있는 반면, 파란색 그래프를 보면 실제 $log p(x)$의 값과 추정된 값이 서로 일치하는 것을 보았을 때 unbiased estimator로 추정을 하였음을 확인할 수 있습니다.

<img src="https://user-images.githubusercontent.com/76925973/200765508-1f58a9ee-e744-460d-92cc-38b1e77b283c.png"  width="400" >

[출처] Residual Flow [^1]


##### 3.2 Memory-Efficient Backpropagation

$$
log(p(x)) = log(p(f(x))) + \mathbb{E}[\boxed{\color{blue}{\sum_{k=1}^{n}}}\frac{(-1)^{k+1}}{k}\frac{v^T[\boxed{\color{red}{J_g(x)}}]^kv}{\mathbb{P}(N\geq k)}]  \cdots (3.2.1)
$$

이와 같이 $log p(x)$를 추정한 unbaised estimator를 이용하여 모델을 학습시킬 때 backpropagation과정에서 메모리를 효율적으로 관리하는 것 역시 중요합니다. 위 식에서 첫번째 박스(파란글씨)에서 n번의 계산을 해야하고, 두번째 박스(빨간글씨)에서 m개의 residual block을 계산해야하기 때문에 위의 식을 그대로 backpropagation에 이용하며 $O(n\cdot m)$ 메모리가 필요하게 됩니다.

따라서 본 논문에서는 메모리를 효율적으로 사용하기 위해서 $log p(x)$의 추정값을 그대로 backpropagation에서 사용하는 것이 아니라 unbiased log-determinatnt gradient estimator를 이용하였습니다. $\mathbb{E}[\displaystyle\sum_{k=1}^{n}\frac{(-1)^{k+1}}{k}\frac{v^T[J_g(x)]^kv}{\mathbb{P}(N\geq k)}]$ 는 $log \vert det(I+J_g(x))\vert$를 추정한 값이므로 log determinant값을 backpropagation에 사용하는 것입니다. 

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

위 식은 log determinant를 미분하는 과정을 보여주는 식으로 (a)는 log값을 미분하는 chain rule을 적용한 것이고, (b)는 determinant 미분을 적용한 식 입니다. (c)는 $(I-f)^{-1} = \displaystyle \sum_{k=0}^{\infty} f^k$ 를 만족한다는 Neumann series를 적용한 것이며, (d)는 Russian Roullete estimator를 (e)는 Skilling-Hutchinson trace estimator를 적용한 결과를 나타낸 것입니다.

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

<img src="https://user-images.githubusercontent.com/76925973/200817001-b285bac3-5a93-4697-a090-67a6c2408460.png"  width="400" >

[출처] Residual Flow [^1]

위 그림을 보면 파란색은 backpropagation을 그대로 진행한 경우를 나타내고 초록색은 unbiased log determinant gradient estimator (3.2.2)식을 이용한 경우, 빨간색은 backward-in-forward (3.2.3)을 이용한 경우이며, 보라색은 두가지 방법을 모두 사용한 경우의 메모리 사용량을 나타냅니다. 이를 통해 두가지 방법을 모두 사용하여 backpropagation을 계산하는 경우 메모리를 훨씬 효율적으로 사용할 수 있음을 확인할 수 있었고 본 논문에서도 역시 두가지 모두를 사용하여 메모리를 효율적으로 사용하고자 하였습니다. 아래 코드는 이를 구현한 것으로 backward과정에서 forward때 계산한 logdetgrad를 이용하여 memory efficient한 계산이 가능하도록 구현하였습니다.

```python
class MemoryEfficientLogDetEstimator(torch.autograd.Function):

    @staticmethod
    def forward(ctx, estimator_fn, gnet, x, n_power_series, vareps, coeff_fn, training, *g_params):
        ctx.training = training
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            g = gnet(x)
            ctx.g = g
            ctx.x = x
            logdetgrad = estimator_fn(g, x, n_power_series, vareps, coeff_fn, training)

            if training:
                grad_x, *grad_params = torch.autograd.grad(
                    logdetgrad.sum(), (x,) + g_params, retain_graph=True, allow_unused=True
                )
                if grad_x is None:
                    grad_x = torch.zeros_like(x)
                ctx.save_for_backward(grad_x, *g_params, *grad_params)

        return safe_detach(g), safe_detach(logdetgrad)

    @staticmethod
    def backward(ctx, grad_g, grad_logdetgrad):
        training = ctx.training
        if not training:
            raise ValueError('Provide training=True if using backward.')

        with torch.enable_grad():
            grad_x, *params_and_grad = ctx.saved_tensors
            g, x = ctx.g, ctx.x

            # Precomputed gradients.
            g_params = params_and_grad[:len(params_and_grad) // 2]
            grad_params = params_and_grad[len(params_and_grad) // 2:]

            dg_x, *dg_params = torch.autograd.grad(g, [x] + g_params, grad_g, allow_unused=True)

        # Update based on gradient from logdetgrad.
        dL = grad_logdetgrad[0].detach()
        with torch.no_grad():
            grad_x.mul_(dL)
            grad_params = tuple([g.mul_(dL) if g is not None else None for g in grad_params])

        # Update based on gradient from g.
        with torch.no_grad():
            grad_x.add_(dg_x)
            grad_params = tuple([dg.add_(djac) if djac is not None else dg for dg, djac in zip(dg_params, grad_params)])

        return (None, None, grad_x, None, None, None, None) + grad_params

```


##### 3.3 LipSwish Activation Function
Backpropagation을 진행할 때 메모리 뿐만 아니라 activation derivative saturation 문제 역시 발생할 수 있습니다. Jacobian을 계산하는 과정에서 일차 미분을 진행하게 되고 gradient를 계산하는 과정에서 이차 미분이 진행되는데, 이때 일차 미분값이 상수가 되면 이차 미분 값이 0이 되면서 gradient vanishing 문제가 발생하는 것입니다. 그래서 논문에서는 립시츠 조건을 만족시키면서 gradient vanishing 문제가 발생하지 않도록 하기 위한 activation function의 두가지 조건을 제시하였습니다.  

- 일차 미분 값이 $\vert \phi'(z)\vert  \leq 1 \text{ } \forall z$   를 만족해야 한다.
- $\vert \phi'(z) \vert$ 가 1과 가까운 점에서의 이차 미분 값이 0이 되어 vanish되서는 안된다.

대부분의 activation function이 첫번째 조건은 만족시키지만 두번째 조건을 만족시키기 어려워 본 논문에서는 두번째 조건을 만족시킬 수 있는 Swish function을 사용했다고 합니다. Swish activaton function은 $f(x) = x \cdot \sigma(\beta x)$ 로 밑의 그림 중 파란색으로 그려진 함수입니다. 그림을 통해서 알 수 있듯이 ReLU의 경우 미분값이 일정해지는 구간이 발생하여 이차 미분 시 gradient vanishing 문제를 야기하지만 Swish 함수의 경우 그렇지 않아 두번째 조건을 만족할 수 있습니다.

<img src="https://user-images.githubusercontent.com/117256746/205494382-ba0da1bd-4ae8-49ff-ba54-4b7dde895a73.png"  width="400" >


그러나 Swish 함수의 경우 일차 미분값이 $\vert \frac{d}{dz}Swish(z)\vert  \lesssim 1.1$ 으로 첫번째 조건을 만족하지 않습니다. 그래서 본 논문은 Swish 함수를 1.1로 나누어 주어 첫번째 조건 역시 만족할 수 있는 LipSwish 함수를 만들었습니다.

$$
LipSwish(z) = \frac{Swish(z)}{1.1} = \frac{z \cdot \sigma(\beta z)} {1.1}
$$

이를 activation function으로 사용하였습니다. 이때 $\beta$ 값은 softplus를 통해 양수를 유지하도록 학습시켰습니다.

<img src="https://user-images.githubusercontent.com/76925973/200821897-be1bf86c-4666-4cd6-9ade-2001592f339f.png"  width="500" >

[출처] Residual Flow [^1]

---

### 4. Experiment

##### 4.1 Mixed Matrix Norm
립시츠 조건을 만족시키기 위해서는 $\vert g(x)' \vert \leq 1$ 이라는 조건을 만족시켜야 합니다. 기존에는 립시츠 조건을 만족시키기 위해서 그 상한선을 0.9로 주어졌는데, 기존 모델과는 달리 Reisdul Flow는 unbiased estimator를 사용하기 때문에 기존의 bias로 인한 안정성 문제를 해결하였기 때문에 더 높은 값을 상한선으로 두어도 립시츠 조건을 만족시키기에 충분했고, 본 논문에서는 0.98을 상한선으로 사용하였습니다.

i-ResNet에서는  조건을 만족시키기 위해 Spectral Norms를 사용했는데, Residual Flow에서는 그에 더해 P-Norms과 Mixed Matrix Norms를 추가로 사용하였습니다.

심층신경망을 사용하는 $g(x)$는 다음과 같이 나타낼 수 있습니다.

$$
z_l = W_lh_{l-1} + b_l, \qquad h_l = \phi (z_l)
$$

여기서 $z_l$은 각 layer의 output을 의미하며, $\phi$는 activation function을 의미합니다. 한 가지 특이한 점은 activation function을 layer를 지난 후가 아니라 지나기 전에 사용했다는 점 입니다.

먼저 spectral norms에서는 다음의 두 가지 방법으로 립시츠 조건을 만족시킵니다.

1. 립시츠 조건을 만족시키는 activation function의 선정, 즉 $\vert \phi '(z)\vert  \leq 1$
2. Weight matrices의 spectral norms으로 bound

$$
\Vert J_g(x) \Vert _2 = \Vert W_L \cdots W_z\phi ' (z_1) W_1 \phi ' (z_0)\Vert_2 \leq \Vert W_l \Vert_2 \cdots \Vert W_2 \Vert_2 \Vert \phi ' (z_1) \Vert_2 \Vert W_1 \Vert_2 \Vert \phi ' (z_0) \Vert_2 \leq \Vert W_l \Vert_2 \cdots \Vert W_2 \Vert_2 \Vert W_1 \Vert_2
$$

즉, $\vert \phi'(z) \vert \leq 1$ 을 만족시키는 $\phi$ 를 사용하며, 각 weight matrices의 spectral norm으로 결과값을 나눠주게 되면 함수 $g(x)$는 립시츠 조건을 만족하게 됩니다.

P-norms에서도 spectral norm과 마찬가지 방법을 사용하였으며, 각 weight를 2-norm이 아니라 p-norm을 통해 계산하였다는 차이점이 있습니다.

1. 립시츠 조건을 만족시키는 activation function의 선정, 즉 $\vert \phi '(z)\vert \leq 1$
2. Weight matrices의 p-norms으로 bound

$$
\Vert J_g(x) \Vert _{\color{red}{p}} = \Vert W_L \cdots W_z\phi ' (z_1) W_1 \phi ' (z_0)\Vert_{\color{red}{p}} \leq \Vert W_l \Vert_{\color{red}{p}} \cdots \Vert W_2 \Vert_{\color{red}{p}} \Vert W_1 \Vert_{\color{red}{p}}
$$

Residual Flow에서는 이 p 값 역시 trainable parameter로 정하여 학습을 통해 결정하였습니다.

Mixed Matrix Norms에서는 각 matrix 값을 자신의 norm로 bound 해주지 않고, 옆의 matrix의 norm으로 교차하여 bound를 해주었습니다. $W_2$는 $\Vert W_1 \Vert$으로, $W_3$은 $\Vert W_2 \Vert$로, $W_L$은 $\Vert W_{L-1} \Vert$로, 다시 $W_1$은 $\Vert W_L \Vert$로 bound 해주어, 각 layer는 립시츠 조건을 만족시키지 못하지만, 전체 network $g(x)$를 보았을 때는 립시츠 조건을 만족시키게끔 하였습니다.

1. 립시츠 조건을 만족시키는 activation function의 선정, 즉 $\vert \phi '(z)\vert  \leq 1$
2. Weight matrices의 mixed matrix norms으로 bound

$$
\Vert J_g(x) \Vert _{\color{red}{p}} = \Vert W_L \cdots W_z\phi ' (z_1) W_1 \phi ' (z_0)\Vert_{\color{red}{p}} \leq \Vert W_l \Vert_{\color{red}{p_{L-1} \rightarrow p_{L}}} \cdots \Vert W_2 \Vert_{\color{red}{p_{1} \rightarrow p_{2}}}  \Vert W_1 \Vert_{\color{red}{p_{1} \rightarrow p}}
$$

Residual Flow에서는 학습된 p으로 mixed matrix norms를 사용하여 0.003 bits/dim의 성능 개선을 보였고 아래와 같이 구현할 수 있습니다.

```python
def normalize_v(v, domain, out=None):
    if not torch.is_tensor(domain) and domain == 2:
        v = F.normalize(v, p=2, dim=0, out=out)
    elif domain == 1:
        v = projmax_(v)
    else:
        vabs = torch.abs(v)
        vph = v / vabs
        vph[torch.isnan(vph)] = 1
        vabs = vabs / torch.max(vabs)
        vabs = vabs**(1 / (domain - 1))
        v = vph * vabs / vector_norm(vabs, domain)
    return v


def normalize_u(u, codomain, out=None):
    if not torch.is_tensor(codomain) and codomain == 2:
        u = F.normalize(u, p=2, dim=0, out=out)
    elif codomain == float('inf'):
        u = projmax_(u)
    else:
        uabs = torch.abs(u)
        uph = u / uabs
        uph[torch.isnan(uph)] = 1
        uabs = uabs / torch.max(uabs)
        uabs = uabs**(codomain - 1)
        if codomain == 1:
            u = uph * uabs / vector_norm(uabs, float('inf'))
        else:
            u = uph * uabs / vector_norm(uabs, codomain / (codomain - 1))
    return u

def compute_one_iter():
    domain, codomain = compute_domain_codomain()
    u = u.detach()
    v = v.detach()
    weight = weight.detach()
    u = normalize_u(torch.mv(weight, v), codomain)
    v = normalize_v(torch.mv(weight.t(), u), domain)
    return torch.dot(u, torch.mv(weight, v))
```

##### 4.2 Density & Generative Modeling

Residual Flow를 다른 flow-based model과 성능을 비교해본 결과 입니다.

<img src="https://user-images.githubusercontent.com/76925973/201391954-32ff92a5-65c8-4c3d-a1c5-7e02ec4a16c3.png"  width="400" >

[출처] Residual Flow [^1]

bits/dim을 비교하였을 때, Residual Flow가 다른 모델들에 비해 MNIST, CIFAR-10, ImageNet32, ImageNet64, CelebA-HQ256의 다섯 가지 데이터셋에 대해 더 좋은 성능을 내는 것을 실험을 통해 확인할 수 있었습니다.

##### 4.3 Sample Quality

다음은 생성된 이미지의 quality를 비교한 결과입니다. temperature annealing을 사용하면 분포의 차이를 극대화 시킬 수 있어서 더 sharp한 이미지를 얻을 수 있지만, 이는 entropy를 희생하는 것이기 때문에 생성된 이미지의 다양성을 감소시켜 본 논문의 저자들은 좋지 않다고 판단해 사용하지 않았습니다.

<img src="https://user-images.githubusercontent.com/76925973/201392100-6c6ba9c4-147c-477d-abc1-94af3f249404.png"  width="700" >

[출처] Residual Flow [^1]

Residual Flow를 통해 생성된 이미지 샘플입니다. 왼쪽 이미지는 CelebA-HQ256 데이터셋에 있는 실제 이미지이며, 오른쪽 이미지는 Residual Flow를 통해 생성된 이미지 입니다.

<img src="https://user-images.githubusercontent.com/76925973/201392242-f4824450-ba49-436e-9927-262a7c071379.png"  width="700">

[출처] Residual Flow [^1]

CIFAR-10 데이터셋의 이미지 및 다른 모델을 통해 생성된 이미지와 함께 비교를 해보았습니다. 비록 PixelCNN이나 Variational Dequantized Flow++로 생성된 이미지들 보다 bits/dim은 오히려 안 좋은 결과를 보여주지만, 저자들은 log-likelihood가 이미지의 퀄리티와 정확하게 매치되는 것은 아니며, Residual Flow를 통해 생성된 이미지가 더 일관된 이미지를 잘 생성한다고 주장합니다.

<img src="https://user-images.githubusercontent.com/117256746/202886256-b976daf9-d93e-4e2e-9a2a-26b9c32ecbb2.png"  >


[출처] Residual Flow [^1]

FID 값을 통해 생성된 이미지를 비교해 보았을 때, DCGAN이나 WGAN-GP와 같은 GAN 기반의 모델들 보다는 떨어지지만, 다른 flow-based model이나 autoregressive model보다 더 좋은 성능을 보이는 것을 확인할 수 있습니다.

##### 4.4 Residual Flow

Residual Flow 모델의 특징을 살펴보면 다음과 같습니다.

- Log-likelihood의 unbiased estimator
- 메모리 효율적인 
- LipSwish activation function 사용

다음의 그래프와 표는 첫 번째와 세 번째 특징에 대한 ablation 실험 결과입니다.

<img src="https://user-images.githubusercontent.com/76925973/201392495-7ab8faa3-58dc-49b7-8053-07baa7cd8484.png"  width="700" >

[출처] Residual Flow [^1]

왼쪽의 그래프를 보면 LipSwhish를 사용했을때 bits/dim이 더 낮아 성능이 좋은 것을 확인할 수 있습니다. 오르쪽의 그래프는 두 가지에 대해 보여주고 있습니다. 첫 번째와 두 번째 행을 보면, 동일하게 ELU를 activation function으로 사용하였을 때 i-ResNet과 Residual Flow의 성능 차이를 보이고 있는데, Residual Flow가 unbiased estimator이기 때문에 더 좋은 성능을 보이는 것을 확인할 수 있습니다. 또한 두 번째와 세 번째 행을 보면, 같은 Residual FLow 모델에 activation function을 ELU와 LipSwish로 변경하며 실험을 진행하였는데, LipSwish를 사용한 모델이 더 성능이 좋은 것을 확인할 수 있습니다.

##### 4.5 Hybrid Modeling

선행 모델인 i-ResNet이 생성모델과 분류기를 동시에 학습시킨 모델임을 고려하여 Residual Flow에서도 hybrid modeling 실험을 진행하였습니다.

주어진 데이터 $x$와 데이터의 라벨 $y$에 대한 확률은 다음과 같이 나타낼 수 있습니다.

$$
log p(x, y) = log p(x) + log p(y \vert x)
$$

여기에서 $log p(x)$는 log-likelihood를 의미하기 때문에 생성모델의 학습을, $log p(y \vert x)$는 주어진 데이터의 라벨 예측을 의미하기 때문에 분류기의 학습을 의미한다고 볼 수 있습니다. Hybrid modeling을 하는 경우, 주로 관심이 있는 쪽은 생성모델의 학습보다 분류기의 학습이기 때문에, $\lambda$라는 1보다 작은 양수의 hyperparameter를 도입하여 weighted maximum likelihood objective[^3]를 최종 objective function으로 사용하며, 이는 다음과 같습니다.

$$
\mathbb{E}_{(x, y) \thicksim p_{data}} \big[\lambda log p(x) + log p(y \vert x) \big]
$$

각 모델의 inference 과정 후에 Multi-layer Perceptron을 통해 분류를 진행하였으며, 그 결과는 다음과 같습니다.

<img src="https://user-images.githubusercontent.com/76925973/201392697-b9823a21-ac7e-4a46-a70d-063a304fd782.png"  width="500" >

[출처] Residual Flow [^1]

$\lambda = 0$일 때는 분류기의 학습만을 진행한 것이며, $\lambda = 1$일 때는 생성모델과 분류기의 중요도를 동일하게 생각한 것 입니다. RealNVP(Coupling) 및 Glow(+ 1 X 1 Conv)와 성능을 비교한 결과 분류기의 학습만을 진행 ( $\lambda = 0$ )했을 때는 Glow의 성능이 조금 더 높았지만, 다른 경우에 대해서는 모두 Residual Flow가 가장 좋은 것을 확인할 수 있습니다.

---
### 5. Conclusion & Application
##### 5.1 Residual Flow

Residual Flow의 장점은 다음과 같습니다.

- 립시츠 조건만을 이용하여 flow-based model을 구성
- log likelihood의 unbiased estimator
- 메모리 효율적인 학습
- 1-Lipschitz 조건을 만족시키는 activation function인 LipSwish 제안
- 일반화된 spectral normalization

하지만 앞선 실험에서 확인한 것처럼 초창기에 제안된 GAN 기반의 모델들 보다 생성된 이미지의 퀄리티가 낮은 것을 확인할 수 있었습니다. 그렇기 때문에 flow-based model이 log-likelihood를 직접 계산할 수 있고, inference 역시 가능하다는 장점을 갖고 있음에도 불구하고 GAN 기반의 모델이 생성모델로서 연구가 더 활발한 것이라고 생각해볼 수 있습니다.

##### 5.2 Application

Residual Flow가 사용된 예시입니다.

- Graph Residual Flow for Molecular Graph Generation[^4]에서는 Residual Flow를 기반으로 분자구조 그래프의 생성모델을 구축하였습니다.

<img src="https://user-images.githubusercontent.com/76925973/201393817-904a50c5-747e-4ecf-9946-c7e3d08ebc16.png"  width="600" >

[출처] Graph Residual Flow for Molecular Graph Generation[^4]

- Hybrid Models for Open Set Recognition2[^5]에서는 Residual Flow의 inference를 통해 out-of-distribution detection을 진행하였습니다. 이는 데이터의 분포를 알아야 하기 때문에 flow-based model을 사용하기에 아주 적합하며, 기존의 다른 flow-based model과 달리 Residual Flow는 unbiased estimator를 구축하였기에 Residual Flow가 가장 적합하게 사용된 사례 중 하나로 꼽을 수 있습니다.
    
$$
pred(x) = \begin{cases}
k + 1 & log p(x) < \tau \\
argmax_{j \in 1, \cdots, k} p(y_j \vert x) &\text{otherwise } 
\end{cases}
$$

<img src="https://user-images.githubusercontent.com/76925973/201393919-600f8123-0d4a-455c-8411-72b4a9a3f5da.png"  width="600">

[출처] Hybrid Models for Open Set Recognition2[^5]


[^1]: Chen, R. T., Behrmann, J., Duvenaud, D. K., & Jacobsen, J. H. (2019). Residual flows for invertible generative modeling. Advances in Neural Information Processing Systems, 32.
[^2]: Behrmann, J., Grathwohl, W., Chen, R. T., Duvenaud, D., & Jacobsen, J. H. (2019, May). Invertible residual networks. In International Conference on Machine Learning (pp. 573-582). PMLR.
[^3]: Nalisnick, E., Matsukawa, A., Teh, Y. W., Gorur, D., & Lakshminarayanan, B. (2019, May). Hybrid models with deep and invertible features. In International Conference on Machine Learning (pp. 4723-4732). PMLR.
[^4]:Honda, S., Akita, H., Ishiguro, K., Nakanishi, T., & Oono, K. (2019). Graph residual flow for molecular graph generation. arXiv preprint arXiv:1909.13521.
[^5]:Zhang, H., Li, A., Guo, J., & Guo, Y. (2020, August). Hybrid models for open set recognition. In European Conference on Computer Vision (pp. 102-117). Springer, Cham.
