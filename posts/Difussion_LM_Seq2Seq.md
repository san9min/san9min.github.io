---
title: "Difussion : Difussion LM & Seq2Seq"
date: 2023-02-26
readingTime: 20 
thumbnail: /images/Difussion_LM_Seq2Seq/thumb.png
tags: [Generative AI, Diffusion, DDPM]
category : [Tech Review]
---

# Diffusion LM Improves Controllable Text Generation

## Diffusion LM

[Diffusion-LM Improves Controllable Text Generation](https://arxiv.org/abs/2205.14217)

https://github.com/xiangli1999/diffusion-lm

- non-autoregressive language model based on continuous diffusions
- embedding from discrete space to continuous space
- continuous diffusion model to discrete text
- Transformer based

`Gaussian noise 벡터들의 sequence에서 시작`해 (inference) denoising을해서 wrods에 corresponding하는 vectors들을 얻는 과정을 거친다.

*a sequence of Gaussian noise vecotrs -- `denoise` —> vectors corresponding to words*

이런 식으로 진행하면 중간에 hierarchy가 있는 contiuous latent variable들이 생긴다. 덕분에 우리가 하듯 gradient 를 때려서 우리가 하던 대로 하면 된다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/269add55-95f8-4bed-9ca2-e838de76ebb4/Untitled.png)

diffusion 모델이 text에 적용하기 힘든이유는 diffusion model은 continuous domain에서 많은 발전을 이뤄왔는데 비해 text가 자체적으로 dicrete한 특성을 갖고 있기 때문. 그래서 이런 standard diffusion에 text를 적용시키기 위해 discrete한 text를 continuous한 domain으로 잘 mapping을 해야한다. NLP에 word to vector가 떠오른다. 본 논문에서는 이를 위해 2가지 step을 추가적으로 적용한다.

1. **`embedding step`** : 같이 학습시켜버림
2. **`rounding step`** with softmax

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e320eedb-ae89-41f4-964c-36be65ca8e6c/Untitled.png)

**Diffusion Models for Text** 는 크게 두가지가 있다 볼 수 있다.

1. text diffusion models on discrete state spaces
    
    discrete data(각 토큰)에 corruption을 주는 식으로 forward process 진행
    
2. `continuous한 latent` variables을 뽑아서 진행

왼쪽에서 오른쪽으로 atuoregressive하도록 generation을 하는 transformer 같은 방식의  language 모델들이 많다.

$p_{lm}(w) = p_{lm}(w_1)\Pi_{i=2}^np_{lm}(x_i|x_{<i})$ ⇒ 지금까지 만들어낸 token들의 sequence를 기반으로 다음 token을 예측하고 이를 eos까지 반복한다. 이렇게 generation의 순서를 고정시키는 것은 모델을 컨트롤하기가 어렵다. 

 Plug and Play Controllable Generation 은 모델 자체는 frozen시키고 외부에 classifier를 따로두는 방식이라 보면된다.

Text Generation은 모델을 학습시켜 모델이 discrete words $w_i$들의 sequence ***w***를 뽑아내는 것 $p_{lm}(w)$ 이고 이것이 controllable하다는 것은 결국 conditional distribution $p(w|c)$ 을 잘 학습한다는 것이다. 본 논문에선 bayes rule로 식을 다시써 외부에 classifier $p(c|w)$ 를 따로두는 방식을 택했다.

### Continuous Diffusion Langauge Modeling

1. **End-To-End Training**

우선 가장 먼저해야될 것은 각각의 `단어들을 d차원의 벡터로 embedding`을 시켜야한다. 그래서 d x n의 텐서를 얻는다.

$EMB(w_i)$ → $EMB(w) = [EMB(w_1),\cdots, EMB(w_n)] \in \mathbb R^{nd}$ 

여기서 diffusion model의 parameter들과 word embedding을 학습할 수 있는 full objective를 제시한다.( Gaussian embedding이나 pre-trained word embedding도 실험을 해봤는데, Diffusion LM에 대해서 `fixed embeddings은 end-to-end training에 비해 suboptimal`에 도달했다고 한다.)

이를 식으로 표현하면 $q_{\phi}(x_0|w) = N(EMB(w),\sigma_0 I)$이고 standard diffusion process를 돌리기 전에 Markov chain하나를 추가해 $w→ x_0$ 로 가게 만든다. 그럼 역으로 $x_0 → w$ 로 가는 step도 필요할테고 이를 `rounding step`이라 하고 식으로 쓰면 $p_{\theta} (w|x_0) = \Pi_{i = 1}^n p_\theta (w_i|x_i)$ 이고 각 단어의 대응하는 벡터들의 `softmax를 이용`해 적절한 토큰을 복구하는 식이다.

이 과정의 objective를 식으로쓰고 simple form으로 쓰면

$$
L^{e2e}_{simple}(w) = \mathbb E_{q_{\phi}(x_{0:T}|w)} [L_{simple}(x_0) + ||EMB(w) - \mu_\theta(x_1,1)||^2 - \log p_\theta(w|x_0)] 
$$

이 된다.

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e6f4dc33-e9f4-4f1c-9c52-ac01d7c0a573/Untitled.png)

1. **Reducing Rounding Errors**

$x_0 → discrete \;\;text$ 은 $x_0$가 word들의 embedding에 정확히 있다면 softmax를 사용해 확률이 가장 높은 토큰으로 복구할 수 있다. 그런데 실제 해보니 잘 안됐고, 이유는 $L_{simple}(x_0)$가  $x_0$와 word사이의 관계를 t가 0근처일때만 관심을 가져서 $x_0$의 structure를 잘 모델링하는데 충분하지 않아서 이다.그래서 모든 t에대해서  $x_0$를 modeling할 수 있도록 loss를 수정했다.

$$
L_{x_0 simple}^{e2e} (x_0) = \sum_{t=1}^T \mathbb E _{x_t}||f_{\theta}(x_t,t)-x_0||^2
$$

`$f_\theta(x,t)$  predicts $x_0$ directly`

그리고 decoding할 때도 이를 사용 ( clamping trick : f가 예측한 $x_0$를 $x_0$와 가장 가까운 word embedding sequence로 mapping, 즉 w_pred)

$x_{t-1} = \sqrt {\bar \alpha}f_\theta (x_t,t) + \sqrt{(1-\bar \alpha)}\epsilon$ — `clamping trick` -→ $x_{t-1} = \sqrt {\bar \alpha}Clamp(f_\theta (x_t,t)) + \sqrt{(1-\bar \alpha)}\epsilon$

### Decoding and Controllable Generation with Diffusion-LM

1. Controllable Text **Generation**

$$
\nabla _{x_{t-1}} \log p(x_{t-1}|x_-t,c) = \nabla _{x_{t-1}} \log p (x_{t-1}|x_t) + \nabla _{x_{t-1}} \log p(c|x_{t-1})
$$

즉 외부에 classifier를 따로 둔채로 학습하겠단 이야기이고, 각각의 step마다 gradient step을 밟겠단 얘기. performance와 speed를 위해 두가지를 수정했는데

1. Fluency Regularization
    
    $\lambda$라는 hyperparam을 도입해 fluency와 control간의 tradeoff를 조절
    
2. multiple gradient steps
    
    control quality를 향상시키기 위해 한번의 diffusion step당 3번의 Adagrad update를 적용
    
1. Minimum Bayes Risk **Decoding**
    
    negative BELU (Bilingual Evaluation Understudy)를 이용해 가장 낮은 BELU score를 기록한 sample을 택한다.
    

---

# DiffuSeq **: Sequence to Sequence Text Generation with Diffusion Models**

## DiffuSeq

[DiffuSeq: Sequence to Sequence Text Generation with Diffusion Models](https://arxiv.org/abs/2210.08933)

https://github.com/Shark-NLP/DiffuSeq

- Conditioning
- Classifier free guidance
- Seq2Seq

Transformer를 model로 사용 ⇒ `Attention`을 통해 classifier free guide 방식을 제시

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/35ff4bdc-8eee-4374-8755-3694e88498df/Untitled.png)

Notation)

**source sequence $w^x =[w^x_1,w^x_2,...w_m^x]$**

**target sequence $w^y = [w_1^y,w_2^y,...,w_n^y]$**

모델은 source sequence의 conditioning으로 target seqeunce를 생성

### Forward Process with Partially Noising

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/314b0108-6a1c-42f1-9666-7a285cbadda5/Untitled.png)

Diffusion-LM에서 처럼 $EMB(w)$을 이용해 discrete text w 를 continuous space로 mapping ( `embedding`, `concatenation`)

- sequence $w^x$ 와 $w^y$의 pair가 주어지고, DiffuSeq는 unified feature space를 학습함

original Markov chain에 $q_{\phi}(z_0|w^{x\bigoplus y})= N(EMB(w^{x\bigoplus y}),\beta_0I)$를 추가한 후 diffusion process 진행

그런데 전체 $z_t$ ($x_t$와 $y_t$ 둘다) 가 아닌 타겟 소스인 $y_t$에만 noise를 추가하는 식으로 forward process → `partially noising`

<aside>
🪧 implementation ) $x_t$까지 같이 corruput 시키고 이를 $x_0$로 replace
training과 inference할 때 둘다 이 방식을 사용

</aside>

### Reverse Process with Conditional Denoising

**Goal :** $z_t$ — denoise —> $z_0$

$p_{\theta}(z_{0:T}) = p(z_T)\Pi_{t=1}^T p_\theta(z_{t-1}|z_t)$ 

**model :** $f_\theta (z_t,t)$ ; Transformer ⇒ $x_t$와 $y_t$ 간의 semantic relation을 학습

$p_{\theta}(z_{t-1}|z_t) = N(z_{t-1};\mu_\theta(z_t,t),\sigma_\theta(z_t,t))$

**Loss**

$$
min_\theta[\sum_{t=2}^T||y_0-\tilde f_\theta(z_t,t)||^2 + ||EMB(w^y)-\tilde f_{\theta}(z_1,1)||^2 + R(||z_0||^2)]
$$

틸다는 f 가 추론한 z중 y만 쓰겠다는 표기

**First term**

$y_0$에 관해서 loss를 계산했기에 $x_0$와 무관해 보이는데 transformer의 attention mechanism 때문에 $x_0$도 고려된 term임

**Third term**

embedding learning을 regularize

<aside>
💡 embedding function을 source와 target이 공유하니까 두개의 feature space를 함께 배움

</aside>

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/238b4eab-8c28-4b55-bd3c-d14d78cc6ce3/Untitled.png)

decoding할 때 Diffusion LM 처럼 BELU score 사용