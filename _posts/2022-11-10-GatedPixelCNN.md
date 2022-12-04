---
layout: post
title: Conditional Image Generation with PixelCNN Decoders
date:   2022-11-10
author: Jeongkee Lim, Doyoung Kim
categories: ["Autoregressive Models"]
tags: generative_model
use_math: true
published: true
img_path: "/assets/img/posts/2022-11-02-GatedPixelCNN"
---

> Van den Oord, Aaron, et al. "Conditional image generation with pixelcnn decoders." Advances in neural information processing systems 29 (2016).

# Introduction
이 논문은 기존의 PixelRNN과 PixelCNN을 제시한 논문[^1]을 발전시켰습니다.
해당 논문에서는 두 가지 형태의 PixelRNN\(Row LSTM, Diagonal BiLSTM\)과 PixelCNN을 제시합니다. 각 모델에 대한 자세한 내용은 해당 논문을 참조하시길 바랍니다. 여기서는 각 모델을 장단점을 바탕으로 어떻게 향상시켰는지에 대해 설명하겠습니다.

PixelRNN은 일반적으로 좋은 성능을 보여줍니다. 하지만, PixelCNN에 비해 학습이 느리다는 단점을 가지고 있습니다. 이는 PixelCNN이 병렬화가 가능하다는 장점 때문에 더 빠르게 학습킬 수 있습니다. 하지만, PixelCNN의 경우 blind spot이 존재한다는 단점이 있습니다. 이 때문에 PixelRNN에 비해 성능이 떨어지는 문제를 가지고 있습니다.

여기서는 두 모델의 장점을 결합한 Gated PixelCNN을 제안합니다. 이 모델이 학습하는데 걸리는 시간을 PixelRNN 대비 절반 이하로 줄였으며 성능 또한 그 이상이라고 합니다.

또한, 저자들은 Gated PixelCNN에 latent 벡터 임베딩을 조건으로 추가한 Conditional PixelCNN을 제안합니다. Conditional PixelCNN은 원-핫 엔코딩 형태로 조건을 주어 여러 클래스로부터 이미지를 생성하는데에 사용될 수 있습니다. 이외에도, 이미지의 high level 정보를 가지고 있는 임베딩을 사용하여 비슷한 특징을 가지고 있는 여러 다양한 이미지를 생성할 수 있습니다.

# Gated PixelCNN
저자들은 두 종류의 convolutional network stack을 결합하여 blind spot을 제거하였습니다.

<img src="https://user-images.githubusercontent.com/60708119/200168243-1085adb4-34b1-4906-91c4-26ad22d510c0.png" style="max-width: 50%">

1. Vertical
  - 현재 픽셀 위에 있는 모든 행에 대한 정보를 가져옵니다.
  - Horizontal Stack에 추가적인 정보로 다시 들어갑니다.
1. Horizontal
  - 현재 행에 대한 정보를 가져옵니다.

이를 통해, blind Spot을 없애고 PixelRNN과 같이 이전 모든 픽셀의 정보를 가져올 수 있습니다.

<img src="https://user-images.githubusercontent.com/60708119/200168422-662533b6-a580-4505-96c3-ebd78e5ca7df.png" alt="1_o4T2iVy6-YcXoxOF51-FEw" style="max-width: 75%"> 

## Gated Activation

$$\mathrm{y} = \tanh(W_{k,f}*\mathrm{x})\odot \sigma(W_{k,g}*\mathrm{x})$$

이때 $*$는 합성곱 연산자, $\odot$은 element-wise multiplication 연산자, $k$는 layer의 인덱스, $f$는 필터, $g$는 게이트로 학습 가능한 합성곱 필터를 의미합니다.

PixelRNN의 LSTM cell 안에 사용되는 multiplicative units -> complex interaction을 모델에 반영할 수 있습니다.   
PixelCNN에도 Gated Activation을 사용하도록 디자인하였습니다.

## Single layer block in a Gated PixelCNN 
### Process 1: Calculate the vertical stack features maps

<img src="https://user-images.githubusercontent.com/60708119/200169369-1e972f87-b12c-464e-b2dd-92755ca4cc00.png" alt="process1" style="max-width: 75%"> 

Vertical stack의 입력은 vertical 마스크가 있는 3X3 합성곱 층에서 처리됩니다. 특성 맵의 결과는 gated activation unit을 통과하고 다음 vertical stack의 입력이 됩니다.

### Process 2: Feeding vertical maps into horizontal stack

<img src="https://user-images.githubusercontent.com/60708119/200169417-e66d74bf-330c-4e18-a12d-91d5f5162d09.png" alt="process2" style="max-width: 75%">   


Vertical stack 특성 맵은 1X1 합성곱 층에 의해 처리됩니다. 

### ❗Before explain process 3 we should think about something

이제, vertical과 horizontal stack의 정보를 결합하는 것이 필요합니다. Vertical stack은 horizontal layer의 입력 중 하나로도 사용됩니다. Vertical stack의 각 convolutional step의 중심은 analysed pixel에 해당됩니다. 따라서 그냥 vertical information을 추가하면 안됩니다. 이것은 future pixel의 정보가 vertical stack의 값을 예측하는데 사용될 수 있기 때문에 autoregressive model의 causality를 깨뜨립니다.

<img src="https://user-images.githubusercontent.com/60708119/200169860-94cb73b3-32bb-4fc2-82f0-ffa9ffc386a2.png" alt="causality" style="max-width: 50%">   

따라서 vertical information을 horizontal stack에 전달하기 전에 padding과 cropping을 이용하여 이동시킵니다. 이미지를 zero-padding하고 이미지 하단을 자르면 vertical과 horizontal stack간의 causality가 유지되도록 할 수 있습니다.

### Process 3: Calculate horizontal feature maps

<img src="https://user-images.githubusercontent.com/60708119/200169930-83637fc4-9fd1-4f18-94e6-acde25df3c07.png" alt="process3" style="max-width: 75%">  

Vertical stack에서 나온 특성 맵을 horizontal convolution layer의 결과와 더해줍니다. 특성 맵은 gated activation unit을 통과합니다. 이 출력을 모든 이전 픽셀의 정보를 고려하는 ideal receptive format을 갖습니다. 

### Process 4: Calculate the residual connection on the horizontal stack

<img src="https://user-images.githubusercontent.com/60708119/200170159-5105fd14-2dd9-4c45-b739-b16038385954.png" alt="process4" style="max-width: 75%">  

Residual connection은 이전 단계(processed by a 1X1 convolution)의 출력을 결합합니다. 신경망의 첫 번째 block에는 residual connection이 없고 이 단계를 건너뜁니다.

다음은 vertical stack과 horizontal stack의 계산에 대한 코드입니다.

```
def get_weights(shape, name, horizontal, mask_mode='noblind', mask=None):
    weights_initializer = tf.contrib.layers.xavier_initializer()
    W = tf.get_variable(name, shape, tf.float32, weights_initializer)

    '''
        Use of masking to hide subsequent pixel values 
    '''
    if mask:
        filter_mid_y = shape[0]//2
        filter_mid_x = shape[1]//2
        mask_filter = np.ones(shape, dtype=np.float32)
        if mask_mode == 'noblind':
            if horizontal:
                # All rows after center must be zero
                mask_filter[filter_mid_y+1:, :, :, :] = 0.0
                # All columns after center in center row must be zero
                mask_filter[filter_mid_y, filter_mid_x+1:, :, :] = 0.0
            else:
                if mask == 'a':
                    # In the first layer, can ONLY access pixels above it
                    mask_filter[filter_mid_y:, :, :, :] = 0.0
                else:
                    # In the second layer, can access pixels above or even with it.
                    # Reason being that the pixels to the right or left of the current pixel
                    #  only have a receptive field of the layer above the current layer and up.
                    mask_filter[filter_mid_y+1:, :, :, :] = 0.0

            if mask == 'a':
                # Center must be zero in first layer
                mask_filter[filter_mid_y, filter_mid_x, :, :] = 0.0
        else:
            mask_filter[filter_mid_y, filter_mid_x+1:, :, :] = 0.
            mask_filter[filter_mid_y+1:, :, :, :] = 0.

            if mask == 'a':
                mask_filter[filter_mid_y, filter_mid_x, :, :] = 0.
                
        W *= mask_filter 
    return 
```

## Conditional PixelCNN

$$p(\mathrm{x|h})=\prod_{i=1}^{n^2}p(x_i|x_1, \space \dots,x_{i-1}, \mathrm{h})$$

저자들은 $\mathrm{h}$에 관한 항을 추가하여 조건부 분포를 모델링하였습니다.

$$ \mathbf{y} = \tanh(\mathbf{W}_{k,\ f}\ * \mathbf{x}\ + \mathbf{V}_{k,\ f}^T\mathbf{h}) \odot \sigma(\mathbf{W}_{k,\ g}\ * \mathbf{x}\ + \mathbf{V}_{k,\ g}^T\mathbf{h})  $$

만약, $\mathrm{h}$가 특정 클래스에 대한 원-핫 엔코딩이라면 이는 모든 layer에 class dependent bias를 추가하는 것과 같습니다. 이러한 조건은 이미지에서 픽셀의 위치에 대한 정보가 아니라는 점을 주목하세요. 즉, $\mathrm{h}$는 이미지의 어디인지가 아닌 이미지가 무엇인지에 대한 정보만을 포함하고 있습니다.

또한, 저자들은 위치를 조건으로 활용하는 함수를 개발하였습니다. 이는 이미지에서 어떠한 구조의 위치에 대한 정보가 임베딩된 $\mathrm{h}$를 활용하는데 유용합니다. $\mathrm{h}$를 deconvolutional 신경망 $m()$를 통해 spatial representation $s=m(\mathrm{h})$로 맵핑함으로써, 아래와 같은 식을 얻을 수 있습니다.

$$ \mathbf{y} = \tanh(\mathbf{W}_{k,\ f}\ * \mathbf{x}\ + \mathbf{V}_{k,\ f}\ * \mathbf{s}) \odot \sigma(\mathbf{W}_{k,\ g}\ * \mathbf{x}\ + \mathbf{V}_{k,\ g}\ * \mathbf{s})  $$

이 때, $V_{k,g}*\mathrm{s}$는 마스킹되지 않은 1x1 합성곱입니다.

## PixelCNN Auto-Encoders
Conditional PixelCNN은 multimodal image distributions $p(\mathbf{x}|\mathbf{h})$ 조건을 줄 수 있기 때문에 Auto-Encoder에 decoder로 사용될 수  있습니다.  
일반적인 Auto-Encoder에서 decoder 부분을 conditional PixelCNN으로 바꾸고 end-to-end로 학습시킵니다.  
그렇게된다면 기존 Auto-Encoder 부분의 Encoder output은 $\mathbf{h}$ 가 되고, Conditional PixelCNN(Decoder)이 $\mathbf{h}$를 condition으로 해서 이미지를 reconstruction합니다.  
PixelCNN은 강력한 unconditional generative model이기 때문에 reconstruction 성능이 좋아질 것으로 기대할 수 있습니다.  
저자들은 이렇게 함으로써 low level pixel statistic을 PixelCNN에서 학습하고, Encoder output인 $\mathbf{h}$ 에서 high-level abstract information을 뽑아내는 것을 기대할 수 있다고 합니다.


# Experiments
## Unconditional Modeling with Gated PixelCNN
먼저, CIFAR-10 데이터 세트에 대해 Gated PixelCNN의 성능을 비교 분석하였습니다.


<img src="https://user-images.githubusercontent.com/26114165/201047261-268a32a9-743b-471a-bc35-c75c01a182b6.png" alt="Table 1" style="max-width: 75%">
*Table 1: CIFAR-10에 대해 여러 모델의 bits/dim(낮을수록 좋음) 성능, 괄호 안의 내용은 훈련할 때의 성능*

Gated PixelCNN은 기존의 PixelCNN 보다 0.11 *bits/dim* 낮은 수치를 보여주며, 생성된 샘플의 시각적 품질에 상당한 영향을 주었습니다. 이는 PixelRNN과 거의 비슷한 수준의 성능을 보여주고 있습니다.

<img src="https://user-images.githubusercontent.com/26114165/201047281-8db6ba45-261f-4a99-9241-bba0c54fc5d4.png" alt="Table 2" style="max-width: 75%">
*Table 2: ImageNet에 대해 여러 모델의 bits/dim(낮을수록 좋음) 성능, 괄호 안의 내용은 훈련할 때의 성능*

그 다음에는 ImageNet 데이터 세트에 대해 Gated PixelCNN의 성능을 비교 분석하였습니다. 여기서 Gated PixelCNN은 PixelRNN보다 더 좋은 성능을 보여줍니다. 저자들은 Gated PixelCNN의 성능이 더 좋은 이유가 PixelRNN이 과소적합 되었기 때문이라고 말합니다. 이렇게 생각한 이유는 일반적으로 모델이 클수록 더 좋은 성능을 발휘하고 간단한 모델일수록 더 빠르게 학습되기 때문입니다.

## Conditioning on ImageNet Classes
두 번째 실험은 Gated PixelCNN을 사용하여 ImageNet 데이터셋의 class-conditional 모델링에 대한 것입니다. i번째 클래스에 대한 정보를 원-핫 엔코딩 형태인 $h_i$가 주어졌을 때 $p(x|h_i)$에 근사되도록 모델링하였습니다. 

이 작업을 통해 모델에게 전달되는 정보의 양은 단지 $\det(1000) \approx 0.003$ bits/pixel에 불과합니다. 그럼에도 불구하고, 이미지 생성에 클래스의 라벨을 조건을 주는 것이 log-likelihood에 큰 향상을 줄것이라 기대했지만 실제로는 큰 차이는 없었다고 합니다. 반면에, 저자들은 생성된 샘플들의 시각적인 품질에서 매우 큰 향상을 관찰할 수 있었다고 합니다.

![Figure 3](https://user-images.githubusercontent.com/26114165/201047245-bb05f61d-7843-41c9-8995-80596a529d6d.png)
*Figure 3: Conditional PixelCNN으로부터 생성된 샘플*

Figure 3에서 8개의 서로 다른 클래스에 대한 샘플의 결과를 볼 수 있습니다. 생성된 이미지들은 각 클래스간에 확실히 구분이 가능하며 각 클래스별로 매우 다양한 이미지를 생성해냈습니다. 예를 들어, 이 모델은 서로 다른 각도와 조명 조건들로부터 비슷한 장면을 생성하였습니다. 이외에도 물체, 동물 그리고 배경들이 명확하게 생성되었다고 주장합니다.  

## Conditioning on Portrait Embeddings
저자들이 진행한 다음 실험은 어떠한 latent representation을 조건으로 주는 실험을 진행하였습니다. 먼저, 그들은 face detector를 활용하여 [Flickr](https://www.flickr.com/) 사이트의 이미지로부터 얼굴을 잘라내어 규모가 큰 초상화 데이터 세트를 생성하였습니다. 다만, 이렇게 생성된 데이터 세트의 이미지 품질은 천자만별인데 이는 많은 이미지들이 좋지 않은 환경에서 휴대폰을 통해 찍힌 이미지이기 때문입니다. 이러한 데이터 세트를 통해 학습된 convolutonal network의 top layer의 latent representation을 이번 실험에 사용합니다. 

이 네크워크는 triplet 손실 함수를 사용하여 훈련됩니다. 이는 특정 인물의 이미지 $x$에서 생성된 임베딩 $h$가 다른 사람의 임베딩과는 멀고 동일한 사람의 임베딩과는 같도록 보장합니다.

이후, 튜플\(이미지=$\mathrm{x}$, 임베딩=$\mathrm{h}$\)을 입력으로 하여 모델 $p(x \vert h)$에 근사되도록 훈련하였습니다. 훈련 세트에 있지 않은 사람의 새로운 이미지가 주어졌을 때 $h=f(x)$를 계산하고 해당 인물의 새로운 초상화를 생성합니다.

![Figure 4](https://user-images.githubusercontent.com/26114165/201046970-36871b95-d43e-4201-bf0c-3fca8824a8a0.png)
*Figure 4: **왼쪽**: source image. **오른쪽**: high-level latent representation으로부터 새롭게 생성된 초상화.*

Figure 4를 통해 생성된 샘플을 확인할 수 있습니다. 소스 이미지의 많은 얼굴특징을 임베딩하고 다양한 자세, 조명 조건 등 여러 조건에서의 새로운 얼굴을 만들 수 있다는 점을 볼 수 있습니다.

![Figure 5](https://user-images.githubusercontent.com/26114165/201046888-4c8c9c82-89ee-4b7c-8e54-b5b7c77de536.png)
*Figure 5: 임베딩 공간에서 선형 보간된 결과가 주어졌을 때 PixelCNN에 의하여 생성된 결과. 가장 왼쪽과 오른쪽 이미지는 보간의 끝 점으로 사용됨.*

마지막으로 저자들은 이미지쌍의 임베딩간의 선형 보간[^2]된 결과가 조건으로 주어졌을 때 생성하는 실험을 진행하였습니다. 이 결과는 Figure 5를 통해 확인할 수 있습니다.

## PixelCNN Auto Encoder
이 실험은 오토인코더의 디코더를 PixelCNN으로 사용하여 end-to-end 훈련을 진행합니다. 32x32 크기의 ImageNet 패치를 사용하며 MSE를 통해 최적화하여 오토인코더를 훈련하였습니다. 모델의 bottleneck의 차원은 10 또는 100으로 설정하였습니다.

![Figure 6](https://user-images.githubusercontent.com/26114165/201046763-7f3886c3-a63c-4a5a-8043-e744812321fa.png)
*Figure 6: 왼쪽부터 오른쪽까지: 원본 이미지, MSE 함수로 훈련된 오토인코더에 의해 복원된 이미지, PixelCNN 오토인코더의 조건부 샘플.*

Figure 6는 각 모델로부터 생성된 이미지를 보여줍니다. 
PixelCNN의 디코더와 함께 bottleneck에서 인코딩된 정보인 representaion $h$가 기존의 디코더를 사용한 것보다 질적으로 다르다는 것을 확인할 수 있습니다. 예를 들어, 가장 마지막 행에서 모델이 입력을 정확하게 다시 생성해내는 것이 아니라 다르지만 비슷한 사람이 있는 실내 공간을 생성하는 것을 볼 수 있습니다.

# Conclusion
저자들은 PixelCNN을 향상시킨 Gated PixelCNN과 Conditional PixelCNN을 제안하였습니다. 수직 및 수평 CNN을 통해 receptive field에서의 "blind spot"을 제거하여 기존의 한계를 극복하였습니다. 

1. Gated PixelCNN은 더욱 효율적으로 계산이 가능합니다.
1. Gated PixelCNN은 PixelRNN 이상의 성능을 보여줍니다.
1. Conditional PixelCNN은 클래스에 대한 조건이 주어졌을 때 해당 클래스에 대응되는 다양하고 현실적인 이미지를 이미지를 생성할 수 있습니다.
1. PixelCNN은 오토인코더에서 강력한 디코더로써 사용될 수 있습니다.

# Limitation
하지만, 이러한 PixelCNN도 여전히 많은 한계를 가지고 있습니다.

1. PixelRNN을 압도할 만큼의 성능은 보여주지 못하고 있습니다.
1. 순차적인 구조는 생성이 진행될수록 에러가 커지는 단점을 가지고 있습니다. 
1. CNN 구조의 특성상 이미지의 디테일은 잘 잡아내지만, 전체적인 구조를 잡아내는 것에는 어려움이 있습니다.

이러한 문제를 해결하기 위해 PixelCNN++, PixelVAE 등이 이후에 제안되었습니다.

# Future Work
## Improvements
1. PixelVAE: A Latent Variable Model for Natural Images
: PixelCNN과 VAE를 결합한 모델입니다.

1. PixelCNN++

## Applications
1. WaveNet: A Generative Model for Raw Audio
1. Video Pixel Networks
1. Genrating Interpertable Images with Controllable Structure
1. Language Modeling with Gated Convolutional Networks

---

# Reference
1. https://docs.google.com/presentation/d/1tYkGAnxPviU_HXpNMiSeaYtVKmWnMNMj956mLqXBB2Q/edit#slide=id.g1a9ca21d74_0_6839

1. https://www.slideshare.net/suga93/conditional-image-generation-with-pixelcnn-decoders

1. https://github.com/anantzoid/Conditional-PixelCNN-decoder

1. https://towardsdatascience.com/pixelcnns-blind-spot-84e19a3797b9

#### Reverse Footnote
[^1]: Van Den Oord, Aäron, Nal Kalchbrenner, and Koray Kavukcuoglu. "Pixel recurrent neural networks." International conference on machine learning. PMLR, 2016.
[^2]: linear interpolation
