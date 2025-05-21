---
title: "Difussion : Latent Diffusion"
date: 2023-02-07
readingTime: 20 
thumbnail: /images/difussion_latent_diffusion/thumbnail.png
tags: [Generative AI, Diffusion, DDPM]
category : [Tech Review]
---



## **High-Resolution Image Synthesis with Latent Diffusion Models**

### Stable Diffusion

📄 [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

🔗 https://github.com/CompVis/stable-diffusion


Stable Diffusion의 기반 논문으로, 핵심 개념은 아래와 같다.

- `Latent Diffusion Model`
- `Cross-Attention based Conditioning`
- `text to image`

pixcel이 아닌 latent space에서 diffusion을 수행함으로써 computational cost를 많이 줄였다. 그런데 난 이 모델이 더 매력적인 부분은 cross attention으로 general한 conditioning input을 받아들 일수 있는 구조였다는 점이다. 이로써 text 로도 gudience를 줄 수 있다는 점이 매우 감동이었다..

### 💁 Guided Diffusion이란?

Stable Diffusion 구조를 이해하려면 먼저 Guided(conditional) Diffusion이 무엇인지 알아야한다. 
Sampling 과정에 condition을 넣어서 생성되는 sample들을 의도한 방향으로 유도할 수 있는 방식이다. 
즉 prior distribution p(x)에 condition y를 줘서 p(x|y)를 모델링하는 것이다. Guided diffusion은 diffusion step마다 condition info를 반영하는 방식이라 생각하면 된다.

$$
p_{\theta}(x_{0:T}|y) = p_{\theta}(x_T)\Pi_{t=1}^Tp_{\theta}(x_{t-1}|x_t,y)
$$

이 수식 구조 덕분에 텍스트나 이미지 등의 외부 입력을 condition으로 줄 수 있다.


---


### 🔻 Gradient 관점의 Guidance

Diffusion model은 SDE(즉, 확률 흐름)로 표현할 수 있다.

그래서 guided diffusion model은 $\nabla \log p_{\theta}(x_t|y)$를 학습해야 한다.

Bayes Rule을 적용하면 아래처럼 분리된다.

$$
\nabla_{x_t} \log p_{\theta}(x_t|y) = \nabla_{x_t} \log (\frac{p_{\theta}(y|x_t)p(x_t)}{p_{\theta}(y)}) \\ = \nabla_{x_t} \log p_{\theta}(x_t) + \nabla_{x_t} \log p_{\theta}(y|x_t)
$$

여기서 두 번째 항에 스칼라 가중치 γ를 곱한 것이 바로 **Classifier Guidance**이다.

$$
\nabla_{x_t} \log p_{\theta}(x_t|y) =\nabla_{x_t} \log p_{\theta}(x_t) + \gamma \cdot\nabla_{x_t} \log p_{\theta}(y|x_t)
$$


---

### 🌀 Classifier Free guidance

📄 [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)

Classifier를 별도로 사용하지 않고 같은 모델로 conditional과 unconditional을 모두 학습하는 방식이다.
Bayes Rule을 다시 적용하면 아래와 같은 형태로 정리된다.

$$
\nabla_{x_t} \log p_{\theta}(x_t|y) = (1-\gamma)\nabla_{x_t} \log p_{\theta}(x_t) + \gamma \cdot\nabla_{x_t} \log p_{\theta}(x_t|y)
$$

이 개념은 Stable Diffusion에서도 다음과 같이 활용된다.


$$\tilde{\epsilon}(z_\lambda,c) = (1+w)\epsilon_\theta(z_\lambda,c) - w\epsilon_\theta(z_\lambda)$$

<aside>
여기서 unconditional term은 0을 embedding 하는 방식 등으로 처리해서, `하나의 모델`로 모두 다룰 있도록 `conditional과 unconditional 을 동시에 학습` 한다.

</aside>

### ⭐ Latent Diffusion의 conditioning  

> `Classifier free guidance`

Latent Diffusion에서는 다음과 같은 방식으로 conditional noise prediction을 계산한다.

<figure class="eq">
$$\epsilon_\theta(x_t,c) = s\epsilon_{cond}(x_t,c) + (1-s)\epsilon_{cond}(x_t,c_u)$$
</figure>


여기서 $c_u$는 empty prompt에 대한 조건(conditional embedding)이고, $s$는 guidance strength이다.

### 🏗 Architecture

![Untitled](/images/difussion_latent_diffusion/01.webp)

`CLIP Text`

- Text understanding component for Text Encoding
- Transformer language model

`U-Net` + `Scheduler`

- Information creator (핵심 diffusion 연산)
- latent space ⇒ faster

`Autoencoder Decoder`

- Image Decoder : 최종 latent ⇒ 이미지 복원

### 🚪 나가며

단순한 T2I 모델을 넘어서,
텍스트라는 일반적인 조건을 활용해 효율적으로 고해상도 이미지를 생성하는 Latent Diffusion의 패러다임을 정립한 논문이다.
