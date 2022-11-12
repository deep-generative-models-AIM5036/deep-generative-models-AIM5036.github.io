---
layout: post
title:  "MADE"
date:   2022-11-13
author: Mose Gu
categories: Midterm
---

# MADE. 

인공지능 분야에서 가장 중요한 영역 중 하나인 Distribution estimation (i.e., 분포 추정)은 데이터의 샘플을 통해 joint distribution p(x)를 추정하는 task이며 이는 데이터의 속성 이해 뿐 아니라 모델 그자체를 파악하는데 사용될 수 있습니다.

기계 학습 분야에서 joint distribution의 specific한 property만의 학습을 통해 많은 task를 수행할 수 있기 때문에 distribution estimation은 인공지능 분야에서 필수불가결 하다고 할 수 있겠습니다.

![그림1](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/c6211bd3-907b-4555-9622-01d835493be0/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221112%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221112T160356Z&X-Amz-Expires=86400&X-Amz-Signature=89fb35557c5507cf3a7eb133f308560e8ebe3228df8baa66bebe247d94d90431&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

그림1

그림1과 같이 distribution estimation으로 테스트 샘플로 joint probability $p(x)$를 추정할 수 있습니다.

하지만 뉴럴네트워크에서 복잡한 연산을 하는 딥러닝 모델은 입력데이터의 dimension을 계속해서 부풀리게 되는데요, 이는 결국 데이터가 sparse해져 성능저하를 초래합니다. 이를 curse of dimenionality라고 하는데요 (그림2), 이는 distribution estimation의 크나큰 limitation이라고 할 수 있겠습니다.

![그림2](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/eb15a69f-ea83-490e-98ca-39a1de07cc65/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221112%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221112T160416Z&X-Amz-Expires=86400&X-Amz-Signature=abdca83b2b50a3456ae974ab3834af5d8fa462d85ecb585fff46cc7881121ce8&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

그림2

그렇기에 저자는 distribution estimation은 high dimensional data에 적용이 힘들다고 지적하는데요, 그러면 어떡하죠? distribution estimation은 인공지능 task에서 꼭 필요하지만 high dimensional data에는 적용이 힘들다니..

어떻게 방법이 없을까요?

저자는 여기서 오토인코더를 선택합니다.
오토인코더는 딥뉴럴넷의 하나로 레이블 없이 비지도 학습을 통해 잠재 차원 표현으로 학습하는 신경망입니다. 이러한 잠재 표현은 뉴런의 크기가 입력보다 더 작지만 같은 분포를 가지게끔 학습됩니다.

왜 오토인코더일까?

오토인코더는 입력데이터를 z스페이스로 압축하여 디코딩을 통해 데이터를 복원하는 모델인데요 (그림3), 여기서 주목할점은 바로 압축입니다. Distribution estimation은 dataset의 dimension에 비롯되고 curse of dimensionality는 이러한 high dimensional dataset의 distribution estimation을 방해하니 high dimension을 low dimension으로 변환하여 이러한 task를 해결하는 방법을 제안합니다.

![그림3](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/b1659502-e802-46c6-8ac4-e75f3357cb40/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221112%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221112T160434Z&X-Amz-Expires=86400&X-Amz-Signature=fadd4933c9164ba3dd4b51a053a82725c604450e1b8ad7ae8abab905e465c56f&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

그림3

Auto-Encoder를 통해 high dimensional dataset의 distribution estimation을 추정하는 것 외에도 오토인코더를 선택한 이유가 있습니다. 저자는 또한 오토인코더에 autoregressive property가 있다고 주장합니다.
Autoencoder는 autoregressive model과 같이 지도학습에서 사용되는 ground truth label과 출력값간의 cost를 줄이는 supervised learning이 아닌 neg log likelihood를 minimize하는 동일한 objective를 가져 학습방법이 비슷합니다.
또한, Autoregressive property에서 출력값인 $x_d$는 $x<d$ 로 학습되므로 출력값을 통해  $x<d$ 를 추정할 수 있습니다. 

![그림4](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/b95de1a4-3bb5-4d9a-a856-59f26ee97734/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221112%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221112T160450Z&X-Amz-Expires=86400&X-Amz-Signature=049bf23a91cd91b30291ea0e83c1a7b8525d3c4a84b101193f4d134208aa0613&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

마찬가지로 오토인코더 역시 previous input $x$에 dependent하게 학습하여 $x\hat{}$ 를 출력하도록 학습되니 은닉층의 뉴런들은 과거의 데이터를 받아 미래값을 추정하는 autoregressive한 특성을 지녀 distribution estimation의 구현 feasibility가 있다는 뜻입니다.

![그림5](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/41904925-4a45-407c-a476-dde6586c3d54/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221112%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221112T160504Z&X-Amz-Expires=86400&X-Amz-Signature=61ba121632d1bfc040a84677d9f5edbabefe9f1555153e47ecd9cb344f876ca4&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

하지만 오토인코더의 고질적인 문제가 있어 저자는 이를 지적합니다.

오토인코더의 그림에서 알 수 있듯, 우리는 우리가 가정한 분포 파라미터를 예측하는 출력층을 가지게 될겁니다. 하지만 $p(x)$ 의 분포를 알기위해선 모든 분포곱으로 계산이 되겠지만 (i.e., $p(x) = p(x_1)p(x_2)p(x_3)...p(x_n)$ ) 이는 independence함이 conponent들 간에 가정되었을때 성립됩니다.

만약 conditional probability 라면 어떨까요? conditional probability에서 를 구하기 위해서는 이러한 공식이 성립되어야 합니다.  $p(x) = p(x_1)p(x_2|x_1)p(x_3|x_2,x_1)...p(x_n|x_n-1)$

오토인코더는 fully connected feed forward network로 모든 뉴런들이 서로 연결되어있는데 이는 독립적이지 않으며 동시에 조건부라고 하기엔 조건부로 종속되는 방식을 알 수 없는 문제점이 있습니다. 자세하게 설명하자면, 입력 노드들이 제각각으로 fully connected 되어 있기에 은닉층을 지날때 벡터안의 element들은 순서가 뒤죽박죽 바뀌게 됩니다. 이의 경우 다음층은 랜덤하게 입력을 받아 conditional 하게 순서를 입력받을 수 없게 되는 겁니다.

![그림6](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/aaf0e83a-14f6-4ca4-a006-eadcb0a691ec/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221112%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221112T160521Z&X-Amz-Expires=86400&X-Amz-Signature=cb3235ecdd65f4273c09c846d6158f12b17a5a4520e7e22c9fa0f4a32af92ef8&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

이러한 문제는 이 그림같이 conditional probability를 구할때 joint distribution을 알 수 없습니다. 

connectivity는 conditional dependence를 respect 해야합니다. conditional probability의 특징은 미래는 과거를 보는데 이의 경우 과거가 있지도 않은 미래를 보게끔 설계가 되는 어불성설이 일어나는 겁니다.
이러한 arbitrary ordering과 joint probability limitation을 해결하는 auto encoder의 joint probability distribution을 estimator를 constrain 하는것이 저자의 motivation이라고 할 수 있습니다.

이에 저자는 오토인코더의 autoregressive한 특징을 사용하되 모든 연결층을 사용하지않고 과거층만을 사용하는 masking을 연결층에 적용한 Masked autoencoder distribution estimation, MADE를 제안합니다.
MADE의 method는 간단합니다. 오토인코더를 masking 함으로써 일부 conncection을 terminate 하는것입니다. 잎서말한 ordering의 arbitrary문제를 해결하기 위해 컴포넌트의 랜덤한 배치를 고려한 입력부터 랜덤하게 assign하여 모델이 랜덤한 인풋에 대해 robust한 학습을 하게 합니다.
또한 connected node가 conditional probability에 필요한 노드만을 입력으로 받고 불필요한 노드와의 연결을 끊게끔 connection을 동적으로 마스킹 합니다. 이러한 마스크엔 두가지 종류가 있는데 압축차원으로 들어가는 encoding connection mask와 출력하는 decoding connection mask가 있습니다.

첫번째 인코딩 마스킹은 마스킹된 입력 connection은 feed forward 파라미터가 같거나 클때, 0을 반환하고 작을때 1을 반환하는 treshhold를 가집니다

![그림7](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/b62b1625-3df3-4cf5-8d2b-6fde705fa7cb/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221112%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221112T160539Z&X-Amz-Expires=86400&X-Amz-Signature=5c6316d8803ff26d91fee170fc4ebaf5f78d8df004a686567d530334bbb853f5&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

이의 경우 은닉층을 지날때 뉴런들이 값을 잃지 않습니다.

출력 connection은 연결된 뉴런값이 더 작을때만 1을 받습니다

![그림8](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/605dbe56-0f85-402e-aa2f-0f32c6cadacf/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221112%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221112T160543Z&X-Amz-Expires=86400&X-Amz-Signature=9a21a073a602c1ba5eba40e7b2667a11c7742ad9279f4acce028bb7e40df777f&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

이의 경우 출력층은 conditional probability를 구할 수 있게 됩니다.

MADE의 아키텍쳐는 다음과 같습니다.

![그림9](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/f0d9b91a-a444-4cc0-ae51-845c2dc953a6/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221112%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221112T160548Z&X-Amz-Expires=86400&X-Amz-Signature=d58eb333689ff679c92b22c16ff29098c98817e028719f92693331e10040f8aa&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

논문에서 말하길 Autoencoder의 advantage는 flexibility라고 합니다. 이는 레이어를 깊게 쌓으면 성능이 올라간다는 뜻입니다. 이에 저자는 layer를 하나 더 쌓아서 MADE를 구현한 모습입니다. 

Order agnostic traning의 모습에서 인풋의 order를 random하게 하여 모델이 autoencoder의 random한 연산에 robust하게 conditional probability를 구하게 되었다고 합니다.

Experiment로 저자는 두개의 dataset을 사용했는데요, UCI binary dataset과 Mnist dataset을 사용하였습니다.

Performance로 보아 MADE가 NADE등의 다른 모델들보다 성능이 더 잘 나온것을 볼 수 있었습니다.

![그림10](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/014331e7-3a54-4f3b-8b59-5eec1152be03/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221112%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221112T160552Z&X-Amz-Expires=86400&X-Amz-Signature=45245c50b7b94366b452111c9a6710e6339f16865d3bec611a1961aba005bced&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

은닉층에서 샘플링한 mnist 인데요, 은닉층에서 충분히 log likelihood가 학습되어있는것을 엿볼 수 있습니다.

![그림11](https://s3.us-west-2.amazonaws.com/secure.notion-static.com/3a071e9b-a16d-401d-8dc6-cd9bb766df4b/Untitled.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20221112%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20221112T160557Z&X-Amz-Expires=86400&X-Amz-Signature=546ace9ed35f47babebdc83783553b2ad52d5b2ef8a4c8e7ce3650b76e847fdd&X-Amz-SignedHeaders=host&response-content-disposition=filename%3D%22Untitled.png%22&x-id=GetObject)

정리하자면 MADE는 Autoencoder를 사용하여 입력 벡터 구성 요소의 조건부 확률 분포를 출력합니
다. 조건부 종속성을 달성하기위해 오토인코더의 연결을 마스킹 하였고 agnostic에 대응하기위해 입력의 order을 random하게 샘플링 하였습니다. 

다들 눈치채셨겠지만 MADE의 분포추정방식은 Normalize Flow와 흡사합니다. 분포들의 곱으로 전체 분포를 추정합니다.
