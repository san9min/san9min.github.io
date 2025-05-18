---
title: "StyleGAN"
date: 2023-02-26
readingTime: 20 
thumbnail: /images/StyleGAN/thumb.webp
tags: [Generative AI, Diffusion, DDPM]
category : [Tech Papers Reivew]
---


[A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948)

> **GAN은 implicit하게 train data의 distribution을 학습한다.
Generator는 학습한 distribution을 바탕으로 이미지를 생성할 수 있다.**
> 

### Image style transfer

**Cycle GAN**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9b4b57b1-a26e-4ec7-86ae-68b42b4eb46d/Untitled.png)

X, Y는 Domain (여기선 스타일정도로 보면 됨)

G : X → Y ; mapping function

F : Y → X  ; mapping function

**cycle consistency loss** 

$F(G(X)) \approx X$

$x → G(x) → F(G(x)) \approx x$

$y → F(y) → G(F(y)) \approx y$

Unparied data로도 학습이 가능해졌다, 별도의 label이 없다→ **unsupervised learning**

**Full objective function (Loss)**

$L(G,F,D_X,D_Y) = L_{GAN}(G,D_Y,X,Y) + L_{GAN}(F,D_X,Y,X) + \lambda L_{CYC}(G,F)$

그러나 $L_{GAN} (G,D_Y,X,Y) = \mathbb E_{y~p_{data(y)}}[logD_Y(y)] + \mathbb E_{x~p_{data(x)}}[log(1-D_Y(G(x))]$ 를 보면

두개의 Domain 사이에서 transfer를 하려면 Discriminator 두개와 Generator 두개가 필요한 것을 볼 수있다.

즉, 이를 **여러 domain 사이에서 transfer를 하려면 Generator와 Discriminator 의 개수가 더 많아진다**.

⇒ 여기서 **StarGAN이 등장**한다.

---

### **StarGAN**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9f292fb6-8e23-46b8-8cf7-984c1779a172/Untitled.png)

위에서 지적한바와 같이 우리가 여러 스타일로 transfer를 하고 싶을 때 그거에 맞게 서로를 연결해줄 generator와 discriminator가 필요한데 stargan에서는 이를 하나로 처리한다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/a6bcb3b1-462b-4f2d-b46d-09e414be7b91/Untitled.png)

구분선을 기준으로 왼쪽(a)은 Discriminator, 오른쪽(b,c,d)은 Generator에 대한 설명이다. 

main idea는 **Domain classification**을 도입해 하나의 Generator와 Discriminator를 사용해 domain간 transfer가 가능하게 했다는 것이다. 

**Discriminator는 Real Image만을 이용해(not use Fake) Domain classification을 학습 (→ classification with real image)**하고, 마찬가지로 Real과 Fake를 구분할 수있도록 학습한다.

**Generator**는 CycleGAN의 아이디어와 유사한데, target domain label과 image를 input으로 받고 fake이미지를 생성한다. (b)

이를 다시 Original domain label과 함께 Generator에 넣어서 image를 reconstruct한다. **이 reconstructed image가 우리가 처음 input으로 넣었던 original image가 되도록 학습 (→ Reconstruction loss)**시킨다. (c)

그리고 마찬가지로 D를 속이도록 학습한다.주목할 점은 **G가 D를 fake image의 domain까지 추가로 속여야한다 (→ classfication with fake image)**는 점이다.(d)

---

### StyleGAN

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9dcbbb9f-8983-41dc-9f88-078b1f18df51/Untitled.png)

기존엔 fixed distribution에서 latent code를 뽑아 바로 Generator에 넣어 주었는데 이는 literally black box 였다. StyleGAN에서는 mapping network, AdaIN, Noise를 도입해 이를 어느정도 해소했다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/681b23fd-19f6-494a-aa2e-593a4345f681/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/beca36f5-3d91-43c1-a78f-7a2f11d3b086/Untitled.png)

1. **Mapping Network**

latent code를 random하게 fixed distribution에서 뽑아 G에 넘겨주는게 기존방식

StyleGAN에선 mapping network를 도입해 z를 w로 먼저 mapping한 후 G에 넣어준다

f : z → w 

이렇게 됨으로써 entanglement를 “어느정도” **disentangle**할 수 있게 되었다.

또한 w를 G의 input layer에 바로 넣어주는게 아니라 각 layer에 넣어주면서 style 정보도 우리가 어느정도 알 수 있게 되었다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2412b475-c2f0-46d0-889f-f912322fa657/Untitled.png)

1. **ProgressiveGAN based**

쉽게 말해 낮은 resolution부터 차근차근 만들어간다는 것이다. 여기서 우리가 주목해야 될것은

**낮은 Resolution에서는 조금 더 global한, coarse한, macroscopic한 feature들과 관련되고**

**높은 Resolution에서는 조금 더 local한, fine한, microscopic한 feature들과 관련된다는 것이다.**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/833060c1-48b8-409a-900f-06a3ab6a3f48/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2705e55b-3968-4f9a-bc34-8241f46270cf/Untitled.png)

1. **AdaIN + Noise**

**AdaIN은** feature를 normalize하고 스타일에 관한 정보로 스케일링하는 역할을 한다. 즉 feature 정보는 남기되 기존의 statistics에 관한 정보를 지우고 style에 기반한 새로운 statistics을 따르게 한다.( AdaIN을 해주고 다음 AdaIn을 해주기 전까지 같은 분포, 영역별로 따르는 분포가 다름, feature 정보는 유지해서 넘겨줌).

다음 convolution 에 더 중요한 정보를 넘겨주는 역할로 이해할 수 있다 → **Style을 더해준다!**

**Noise를 주면서 조금 더 detail하고 local한 정보들을 정교하게 생성할 수 있다.**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8d241d98-bc8e-4c9e-899f-22828163cc6f/Untitled.png)

⭐⭐⭐이 Style Mixing은 우리 프로젝트랑 밀접한 관련이 있다. Style Mixing이 가능하고, 어떤 원리에 의해 되는지 이해해야한다.

mapping network를 통과한 w, 즉 style(y = W * **w** +b, style은 단순히 w를 affine transform한 결과이므로)과 관련된 두 벡터(w1,w2)를 믹싱한 결과를 보여준다. 

⇒ low resolution level은 pose, identity, general hair-style, face-shape, eye-glasses 같은 coarse(global,macroscope)한 특징들과 관련있고,

high resolution으로 갈 수록 color 같이 더 fine(local,microscope)한 특징들과 관련이 있음을 볼 수 있었다. (→ 우리가 control할 여지가 보인다…!)

---

**StyleGAN2 는 가져다 사용하면 될 거고 우리 플젝에 사용할 중요한 원리는 StyleGAN1에 있다**

v1에서 AdaIN 할 때 normalize와 modulation하는 과정에서 문제가 있어서 effect는 유지하되 다른 방식으로 style을 더해주는 과정을 진행함

---

**Toonify**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/92d5882a-3083-4361-b1fd-bce273491654/Untitled.png)

StyleGAN을 base로 삼고있다.

우리가 선택한 특정 Resolution을 기준으로 layer block 영역을 두개(낮은 해상도, 높은 해상도)로 나눈다. 하나의 영역에는 base model의 weight를, 다른 영역에는 trasnfer learned model의 weight를 사용한다. 이 논문은 StyleGAN Style Mixing을 이해하면 쉽게 납득 가능하다.

---

**GAN Inversion**

가장 궁금했던 부분이 해결된 지점이다. GAN에는 어떤 distribution(ex. Gaussian)에서 random하게 latent code뽑아 Generator에 넣어줬는데, 그러면 도대체 image를 input으로해서 어떻게 style을 바꾸는 걸까? 에대한 답을 주었다.

G가 어떤 이미지를 생성해 냈을 때, 이 **이미지에 대한 latent vector를 찾는**… 그래서 inversion… WOW…

우리 프로젝트에 가장 중요한 Subtask가 될 것 같다. ⇒ ⭐⭐⭐ **GAN Inversion** 

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/647046c7-4f63-41b1-b922-c9cb4ea13b37/Untitled.png)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e3b7c62d-171a-48d0-ba4a-d1d8d9fddf10/Untitled.png)