---
layout: post
title: Improved Variational Inference with Inverse Autoregressive Flow
---

***
2016년도에 발표된 Improved Variational Inference with Inverse Autoregressive Flow 논문에 대해 소개드리겠습니다.

***

![_config.yml]({{ site.baseurl }}/IAF_IMGs/01.PNG)

***

![_config.yml]({{ site.baseurl }}/IAF_IMGs/02.PNG)

소개 순서는 위와 같습니다. 기본적으로 논문에서 서술하고 있는 순서입니다.

***

![_config.yml]({{ site.baseurl }}/IAF_IMGs/03.PNG)

이 논문은 새로운 타입의 NF를 제안하고 있습니다. inverse autoregressive flow의 약자로 IAF라는 NF입니다.
이 것은 고차원의 latent space로 확장이 용이하다고 합니다.
그리고 이를 VAE에 적용하여 IAF의 효과를 보여주었습니다.

아래는 논문 서두에 있는 그림입니다. prior 분포를 기존의 가우시안 포스테리어로 피팅을 한 경우, 빈공간이 많이 보이는 것처럼
완전히 피팅을 못하였는데요, IAF로 한경우에는 prior와 거의 유사하게 피팅되었습니다. 
이는 IAF가 그만큼 포스테리어의 flexibility를 향상시킨 결과라고 할 수 있습니다. 

***

![_config.yml]({{ site.baseurl }}/IAF_IMGs/04.PNG)

다음은 variational inference에 대한 일반적인 내용입니다. 간단히 이야기하고 넘어가자면,
생성 모델의 목표는 마지널 라이클리후드를 최대화 하는 것인데,
이 마지널 라이클리후드를 계산하는 것이 intractable하여, 
근사 분포인 q를 도입하여 ELBO를 최적화하는 것으로 문제를 푸는 방안이 제안되었습니다.

그러기 위해서는 q가 p와 유사한 분포가 될 수 있을 만큼 flexible하다면, 
KL D는 작아질 것이고, p와 유사한 분포를 얻을 수 있습니다.

다음으로 논문에서는 
효율적으로 ELBO를 최적화 하기 위한 inference 모델의 요구사항으로 아래 두가지를 언급하고 있습니다.
하나는 q를 계산하고 미분하기 쉬워야하며
다른하나는 그 모델에서 샘플링하기 쉬워야 한다고 하였습니다.
왜냐하면 최적화 과정에서 자주 사용되기 때문에 효율적으로 계산할 수 있어야 합니다.

***

![_config.yml]({{ site.baseurl }}/IAF_IMGs/05.PNG)

다음에는 NF에 대해서 이야기 하고 있습니다.
지난 수업시간에 들었듯이, NF는 쉬운 분포로부터 역변환이 가능한 여러 연산을 수행하여 유연한 포스테리어를 얻을 수 있습니다.

가우시안 등 간단한 분포로부터 z0을 선택하여 여러번의 변환과정을 거칩니다.
그러면 마지막 iteration T에서는 원하는 분포를 얻을 수 있게 됩니다.
이 분포는 다음과 같으며, T번의 변환 과정동안 자코비안 디터미넌트를 구할 수 있다면, 
마지막이 분포도 계산할 수 있게 됩니다.

중요한 점은 이 자코비안 디터미넌트를 얼마나 쉽게 구할 수 있느냐 입니다.

***

![_config.yml]({{ site.baseurl }}/IAF_IMGs/06.PNG)

이제부터 본 논문에서 이야기하는 inverse autoregressive 변환에 대한 이야기 입니다.

앞에서 이야기한 자코비안 디터미넌트를 구해야 합니다.

자코비안 메트릭스는 미분한 값들로 이루워진 행렬인데요,
autoregressive 구조로 뮤와 시그마에 대한 자코비안은 대각선이 0인 하삼각행렬이 됩니다. 

좀더 자세히 설명 드리면, 
yi는 뮤와 시그마로 만들어지는데, 미래의 yi로 과거의 뮤와 시그마를 미분하면
이는 yi에 관계가 없는 상수로 취급되어 0이 됩니다.
뮤i, 시그마i 도 yi 보다 과거의 값입니다. 그래서 같은 이유로 대각선

뮤i는 y1부터 i-1까지로 만들어지는 것으로 yi는 뮤i 관점에서 미래에 있는, 즉 관계가 없는 변수입니다.
이것으로 미분을 하면, 상수이므로 0이 됩니다. 

또한, 엡실론0을 선택한다음, 순차 변환을 통해서 yi를 구해갑니다. 즉, yi를 계산할 때 yi-1까지의 값이 필요하며
차원 D 가 커지면 계산은 복잡하게 됩니다.

이 식을 엡실론 기준으로 다시 정리하면 아래와 같이 됩니다. 
이렇게 바꾸면 다음의 두가지 장점이 생깁니다. 자코비안을 계산하기 쉬우면, 병렬화가 가능합니다.
하나씩 알아보겠습니다.

***

![_config.yml]({{ site.baseurl }}/IAF_IMGs/07.PNG)

이 그림은 자코비안에 대한 예시입니다.

***

![_config.yml]({{ site.baseurl }}/IAF_IMGs/08.PNG)

먼저 자코비안 입니다. 
앞에서 이야기한데로, 뮤와 시그마를 y로 미분하면 대각선이 0이 되고 자코비안이 0이 됩니다.
하지만, 식을 엡실론으로 정리한 뒤에 y로 미분하게되면, 같은 인덱스에 대해서도 시그마 분의 1일이 나오게 됩니다.
디터미넌트는 대각선의 곱이므로 오른쪽 아래와 같이 계산할 수 있게 됩니다.

***

![_config.yml]({{ site.baseurl }}/IAF_IMGs/09.PNG)

조금더 정리해보자면 다음과 같습니다.
왼쪽은 autoregressive변환의 경우이며, 오른쪽은 인버스 오토리그레시브 변환입니다.

z는 레이턴트 변수이면, x는 데이터 값입니다.
AT에서는 z를 구하는데, x의 값만을 입력을 받습니다.
반면, x를 구할때는 x와 z값을 같이 입력으로 받습니다. 즉 앞의 x1가 계산이 된 후에 x2를 계산할 수 있습니다.
병렬화하기 어렵습니다.
즉 x에서 z로 학습하는 과정은 빠르나, z에서 x로 샘플링하는 과정은 느립니다

반면 IAF는 역변환을 통해서 z를 만드는데 있어서는 오래 걸리지만, 샘플링은 병렬로 처리 할 수 있게 됩니다. 

***

![_config.yml]({{ site.baseurl }}/IAF_IMGs/10.PNG)

이 inverse autoregressive 변환을 이용하여 IAF를 제안하였습니다.
앞에서 언급된 T번 변환된 q 값은 그림과 같이 초기값과 디터미넌트를 입력하여 풀어주면
아래와 같이 최종 형태를 얻게 됩니다.
즉 D차원에 대해 계산을 반봅하는데, IAF의 스탭 T만큼 디터미넌트를 더해주는 모양입니다.
간단한 모양으로 표현되었습니다.

***

![_config.yml]({{ site.baseurl }}/IAF_IMGs/11.PNG)

처음에 입력된 x와 별도로 엡실론을 선택하여 z초기 값을 만든 후, IAF 스탭을 거치게 됩니다.
여기서 추가적으로 h 값이 만들어지고 각IAF 스텝에 입력됩니다. 
이는 LSTM에서 과거 정보를 어느 정도 유지하기 위해 cell state를 가지는 것과 유사한 이유 같습니다.
하나의 IAF 스탭은 다음과 같이 생겼습니다.

이렇게 만들어진 z, h는 Autoregressive NN에 입력됩니다. 이는 Autoregressive특성을 같는 NN이면 어떤 것이든 사용할 수 있을 것같습니다.
여기서는 MADE를 사용하였다고 합니다. 
거기서 m, s를 만들고, 새로운 zt를 만들때, 과거 z와 m에 비율을 적용하여 더하였습니다. 
이는 LSTM에서 영감받았다고 합니다. 그리고 forget gate bais로 알려진 바와 같이, st값이 1, 2면 값일때 좋은 성능을 내었다고 합니다.
각 IAF에서는 이런 연산을 수행합니다.

***

![_config.yml]({{ site.baseurl }}/IAF_IMGs/12.PNG)

다음은 현재까지 이야기한 내용을 수도코드로 표현한 것입니다.
중간에 보면 앞에서 보았던 zt식이 보이며, 매 스탭마다 디터미넌트 값을 더하는 것을 확인할 수 있습니다.

***

![_config.yml]({{ site.baseurl }}/IAF_IMGs/13.PNG)


***

![_config.yml]({{ site.baseurl }}/IAF_IMGs/14.PNG)

다음은 실험입니다.
resnet블럭을 사용하였으며, autoregressiveNN는 2레이어짜리 MADE 1개로 구성하였다고 합니다.

MNIST데이터의 경우,
여러 포스테리어의 성능을 비교해본 결과 이다. IAF의 VLB가 depth와 width가 커질수록 커지며, 최대 -79.1까지 도달했다. 

사이파텐의 경우, 3.11의 bit per dim을 달성하여 다른 latent variable 모델보다 성능이 좋았으며
기존 라이클리 후드를 사용하는 모델의 최고 성능과 비슷하게 나왔다. 이는 IAF의 스텝을 더 쌓으면 될 것같다고 하였다.

샘플링 속도에서는 0.05초로 pixelCNN의 52초에 비해 압도적으로 빨랐다고 합니다.

***

![_config.yml]({{ site.baseurl }}/IAF_IMGs/15.PNG)

마지막으로 내용을 정리하면, p를 찾아가는 방법으로 q를 도입하였으며, q를 잘 변형하여 새로운 q를 만들어 성능을 높인 논문으로
변환하는 과정에 autoregressive성질을 반영하여 디터미넌튼 연산과 병렬 연산이 가능하도록 하였습니다.

***

![_config.yml]({{ site.baseurl }}/IAF_IMGs/16.PNG)

***

![_config.yml]({{ site.baseurl }}/IAF_IMGs/17.PNG)

***

![_config.yml]({{ site.baseurl }}/IAF_IMGs/18.PNG)

***

![_config.yml]({{ site.baseurl }}/IAF_IMGs/19.PNG)

***

![_config.yml]({{ site.baseurl }}/IAF_IMGs/20.PNG)

***

![_config.yml]({{ site.baseurl }}/IAF_IMGs/21.PNG)

***

감사합니다.

