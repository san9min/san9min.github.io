---
title: "Diffusion : DDPM"
date: 2023-02-17
readingTime: 20 
thumbnail: images/Diffusion_DDPM/thumb.jpg
tags: [Generative AI, Diffusion, DDPM]
category : [Paper Reivew]
---
## Genetrative Model Framework
Genetraion은 크게 두가지 framework으로 나뉜다.
- **Likelihood-based**
    - autoregressive models
    - variational autoencoders
    - flow-based models
    - **`diffusion models`**
- **Implicit model**
    - generative adversarial networks(GAN)

likelihood based 인 diffusion model에 관해 알아보자

![B948109D-1EBD-4BF6-B739-FB8EF24AE95E.png](/images/Diffusion_DDPM/01.png)

---

## Background

Diffusion에 관해 설명하기 전에 알고있어야할 사전지식들을 간단하게 정리해보겠다.

### KL-Divergence

`두 확률 분포가 얼마나 다른지`를 계산(정보 엔트로피 차이를 계산)

$$
D_{KL}(q||p) = \begin{cases} - \sum_i q_ilog\frac{p_i}{q_i} 
&\text{(discrete form)}\\
-\int q(x)log \frac{p(x)}{q(x)} &\text{(continuous form)}
\end{cases}
$$

이고 식을 전개하면 self-entropy term과 cross entropy(or average of negative log-likelihood in p of samples from q)으로 분리할 수 있다.

$$
D_{KL} = -H(q) + H(q,p)
$$

만약 분포가 continuous 한 경우

$$
\int q(x) log(q(x))dx + \int q(x)(-log(p(x))dx\\ = -H(q) + H(q,p)
$$

로 쓸 수 있다.

예를들어, 우리가 q를 p로 근사시키기 위해, KL divergence를 minimize하면 self entropy term은 q의 variance를 증가시키켜 넓게 퍼진 분포가 되려는 경향을 갖게하고 cross entropy term은 p 분포에서 likelihood가 가장 높은 지점에서 Dirac -delta function처럼 되게하려는 경향을 갖게 할 것이다. 

이 두 term을 통해(싸우는 느낌?) q가 p로 근사가 된다.

**KL divergence의 특성** 

항상 0 이상이다. CE는 아무리 낮아져봤자 즉 q와 p가 같은 분포가 된다 했을 때 self-entropy이다. 그러므로 최솟값이 0이고, 절대 음수가 될 수 없다. 

거리 개념이 아니다. 

일반적으로 $D_{KL}(p|q) \neq D_{KL}(q|p)$이다.

### Bayes Rule

$$
P(H|E) = \frac {P(H) P(E|H)}{P(E)}
$$

Straight forward하다. 논문을 읽다보면 확률 용어가 많이 나와 헷갈렸는데 용어를 간단히 정리하자.

**Terms**  

E : Evidence( ~ sample x), H : Hypothesis( ~ latent z)

- P(H) : Prior Probability ( 사전에 알고 있는 H가 발생할 확률 )
- P(E|H) : **Likelihood** of the evidence E if the Hypothesis H is true ( 모든 사건 H에 대한 E가 발생할 likelihood ) ⇒ How well H explains E !
- P(E) : Priori probability that the evidence E itself is true ( E의 사전확률,즉 E가 발생할 확률, marginal이라고도 함 )
- P(H|E) : **Posterior Probability** of ‘H’ given the evidence

### Monte Carlo Method

랜덤 표본을 뽑아(`sampling`을 통해) 함수값을 확률적으로 계산하겠다 이고 결국 근사(`approximation`) 시키겠다는 이야기다.

예를들면

$$
\int p(x)f(x)dx = E_{x \sim p(x)}[f(x)] \approx \frac 1 K \sum_i^K f(x_i), x_i \sim p(x)
$$

확률 밀도함수 p(x)를 따르는 x에대한 f(x)의 기댓값을 구하고 싶다했을 때 p(x)에서 K개의 샘플을 뽑아 이로 계산해도 괜찮다는 이야기

### ELBO : Evidence Lower Bound

결국 **`Variational Inference`**는 사후확률 분포(posterior) `p(z|x)`를 다루기 쉬운 확률분포 `q(z)로 근사`하고 싶은 의지이다.

$q^*(z)  = argmin_{q(z) \in Q} D_{KL}(q(z)||p(z|x))$

![Untitled](/images/Diffusion_DDPM/02.png)

![Untitled](/images/Diffusion_DDPM/03.png)

그러므로 KL divergence를 이용해 이를 식으로 표현하면

$$
D_{KL} (q(z) || p(z|x)) = \int q(z) \log \frac{q(z)}{p(z|x)}dz\\ \text{by Bayes Rule} \\= \int q(z)\log\frac{q(z)p(x)}{p(x|z)p(z)}dz \\= \int q(z)\log\frac{q(z)}{p(z)}dz +\int q(z)\log p(x)dz -\int q(z)\log p(x|z)dz \\ = D_{KL}(q(z)||p(z)) + \log p(x) - E_{z \sim q(z)} [ \log p(x|z)]
$$

로 정리된다. 

bayes rule에 의해 우리는 사후확률 p(z|x)를 p(z), p(x), p(x|z)로 가져갔다. 

여기서 latent variable z 의사전확률 분포 p(z)는 x와 무관해 any kind of distribution이여도 괜찮으므로 가장 간단하고 예쁜 Gaussian이라 하자. 그리고 q를 $\phi$, p를 $\theta$로 parameterize하자. 

그런데 여기서 $\log p_{\theta}(x)$는 intractable하다. 그래서 우리는 tractable한 lower bound(ELBO)를 잡고 이를 maximize하는 방식을 취한다.

Expectation : $D_{KL} (q_{\phi}(z|x) || p(z|x))$를 minimize하는 $\phi$를 찾자

Maximization :  $\phi$를 고정하고 $\log p_{\theta}(x)$의 lower bound를 maximize하는 $\theta$를 찾자

[Expectation-Maximization](https://www.notion.so/image/https%3A%2F%2Fs3-us-west-2.amazonaws.com%2Fsecure.notion-static.com%2F4e8e6c6e-5d0f-4c41-a457-ad6fa203a2a1%2FUntitled.png?table=block&id=1163df0e-ec45-812f-9a80-c4068024117e&spaceId=14f1eeea-15e5-42e2-b9d5-486040ff5c3d&width=2000&userId=df45ccee-de8f-4276-80c1-4933eb8b1e4d&cache=v2)

<aside>
log p(x) 를 evidence라고 한다

 $\theta$로 parameterized된 우리의 model이 observed data x에 대해 marginal probability를 계산했을 때 만약 우리 모델이 잘 학습이 되었다면 높은 값을 내놓을 것이다. 즉, 학습 중에 $\theta$를 잠시 fix해놓고 evaluation을 했을 때 높은 값을 내놓고 있다면 우리는 잘 가고 있다는 것이다. 그래서 $logp(x;\theta)$를 우리가 잘 가고 있다는 의미에서  evidence라 한다. 
</aside>

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4e8e6c6e-5d0f-4c41-a457-ad6fa203a2a1/Untitled.png)

$D_{KL} (q_{\phi}(z|x) || p(z|x))\ge 0$ 이므로 

$$
\log p_{\theta}(x) \ge E_{z \sim q(z)}[\log p_{\theta}(x|z)] - D_{KL}(q_{\phi}(z|x)||p(z)) 
$$

가 된다. 

**ELBO** 

$$
⁍
$$

1st term

$E_{z \sim q(z|x)}[\log p_{\theta}(x|z)]$

`Reconstruction Error` → generative model( Decoder in VAE )

2nd term

$D_{KL}(q(z|x)||p(z)) \text{ or } E_{q(z|x)} [\log \frac {q(z|x) }{p(z)}]$

`Regularization term` → inference model( Encoder in VAE )

### SDE : Stochastic Differential Equation

TODO

---

# **Denoising Diffusion Probabilistic Models**

## DDPM

[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

- parameterized Markov chain
- trained using variational inference
- `learn to reverse a diffusion process`

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1b107a88-b5c8-4fea-80df-50b6b903fdc2/Untitled.png)

Diffusion model은 latent variable 모델이다.

**latent : a hidden continuous feature space**

**GOAL**

$$
p_{\theta} (x_0) = \int p_{\theta}(x_{0:T})dx_{1:T}
$$

$x_1,..x_T$ 는 latent들이고 data $x_0 \sim q(x_0)$와 같은 dimension을 갖는다.

> data에 `noise`를 더해가는 것을 `forward process`
noise로 부터 `de-noise`해나가는 `reverse process`라한다.
> 

우리 model은 이 reverse process를 학습하고 새로운 data를 generation한다.

### Forward Process (diffusion process)

`forward process는 Gaussian noise를 더해가는 과정`

이는 Markov chain으로 formulate할 수 있다. 즉, 

$$
q(x_{1:T}|x_0) = \Pi_{t=1}^Tq(x_t|x_{t-1}) 
$$

by Markov chain with step T

Markov chain은 각 step은 오직 직전 step에만 의존함을 의미한다.

$q(x_{1:T})$는 q를 timestep 1부터 T까지 반복해서 가함을 의미하는 notation

$\text {where}$

$$
⁍ 
$$

각 스텝에서 Gaussian noise를 더한다

여기서 $\beta_t$는 variance schedule이고 I가 identity이므로 각 dimension은 같은 std를 갖는다. 상수로 둬도 되고, 시간에 따른 변수로 두어도 된다. 논문에선 higer t로 갈수록 커지는 방향으로 linear하게 두었는데 다른 논문에선 cosine shcedule 이 잘 됐다고 한다.

$\beta_t$를 이용해 scaling한 후 더해주는 이유는 variance가 diverge하는 것을 막기위함으로 볼 수 있다. 

<aside>
💡 Gaussian noise를 더해가면, 최종 step (time T)에서는standard normal prior $N(x_T;0,I)$ 가되고 그래서 diffusion이라 한 것같다.

</aside>

그런데 여기서 어떤 순간 t ( $0 \le t \le T)$에서 $x_t$를 알고 싶다고한다면, 위의 식을 이용해 반복적인 계산을 수행하면 된다. 그러나 t 가 크다면 이는 좋은 방법이 아닐 것이다.

**Reparameterization trick (for sampling** $x_t$ **at once)**

만약 우리가

$\alpha_t = 1-\beta_t , \bar \alpha_t = \Pi_{s=0}^t \alpha_s$ 라고 잡는다면, t에서 $x_t$를 sampling하는 것을 closed form으로 쓸 수 있을 것이다.

⭐⭐⭐

$$
q(x_t|x_0) = N(x_t;\sqrt {\bar \alpha_t} x_0, (1- {\bar \alpha_t})I)
$$

sample = mean + (var**0.5)*epsilon

⭐⭐⭐

- **Proof )**
    
    let $\epsilon_0, \cdots \epsilon_{t-2},\epsilon_{t-1} \sim N(0,I)$
    
    $x_ t = \sqrt{(1-\beta_t)}x_{t-1} + \sqrt {\beta_t} \epsilon_{t-1} \\ =\cdots  \\=\sqrt{\bar \alpha_t}x_0 + \sqrt{1-\bar{\alpha_t}}\epsilon_0$
    
    증명의 핵심논리는 다른 variance를 갖는 두개($\sigma_1^2,\sigma_2^2$)의 Gaussians을 merge해 새로운 distribution(with variance $\sigma_1^2+\sigma_2^2$ )을 만드는 것
    

좋다. 

확인을 해보면 t → $\infty$로 갈때 $q(x_t|x_0)$가 $N(x_t;0,I)$로 감도 볼 수 있다

이제 우리는 any timestep t에서 noise를 sampling할 수 있게됐고, 이를 통해 $x_t$를 $x_0$와 $\epsilon$의 함수로 볼수 있게 되어 forward process에서 $x_0$만 알면 바로  **$x_t$를 얻을 수 있게됐다.**

$$
x_t = \sqrt{\bar \alpha_t}x_0 + \sqrt{1-\bar{\alpha_t}}\epsilon_0
$$

or

$$
x_0 = \frac{1}{\sqrt{\bar \alpha_t}}(x_t - \sqrt{1-\bar \alpha_t}\epsilon)
$$

### Reverse Process (denoising process)

`reverse process는 neural network이 학습할 과정`

$q(x_{t-1}|x_{t})$를 원하나 어려워서 neural network를 이용

forward process의 Gaussian noise가 충분히작을 때 reverse process 또한 Gaussian이 되고 이는 neural network를 이용해 근사시켜 mean과 variance를 예측하는 모델을 만들 수 있다.

$$
p_{\theta}(x_{t-1}|x_t) = N(x_{t-1};\mu_{\theta}(x_t,t), \Sigma_{\theta}(x_t,t))
$$

Reverse process는 $p(x_T) = N(x_T;0,I)$부터 출발해 (learned) Gaussian trainsition을 하는 Markov chain이다. 즉, trajectory는

$$
⁍
$$

로 fomulate할 수 있다.

<aside>
💡 neural network에 timestep t를 conditioning하면 model은 각 time step의 Gaussian의 mean과 variance를 예측할 수 있게된다.

</aside>

### Training

`ELBO`on the negative log likelihood를 optimize

$$
E[-\log p_{\theta}(x_0)] \le E_q[-\log \frac {p_\theta (x_{0:T})}{q(x_{1:T}|x_0)}] \\ = L
$$

이고 위 식을 정리하면

$$
L = E_q[D_{KL}(q(x_T|x_0)||p(x_T))+\sum_{t>1}D_{KL}(q(x_{t-1}|x_t,x_0)||p_{\theta}(x_{t-1}|x_t))-\log p_{\theta}(x_0|x_1)]
$$

이 된다.

<aside>
💡 $q(x_{t-1}|x_t)$는 intractable하지만 $x_0$의 conditioning을 주면 tractable하다고 함 → generative model이 reverse diffusion step으로 generation을 하기 위해선 reference image $x_0$가 필요하다

</aside>

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d411bb47-dcf7-4b5a-afb1-89eb1e580dcc/Untitled.png)

> 동하 주)
> 
> - 위 식 증명 과정
>     
>     $q(x_{t-1}|x_t)$는 무시가능해짐. (Markov Process이기 때문)
>     
>     ![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c052df61-f676-421a-b570-5f8733d6d4d0/Untitled.png)
>     

결국 하고 싶은 것은 `$p_{\theta}(x_{t-1}|x_t)$와 forward process posteriors $q(x_{t-1}|x_t,x_0)$를 비교`하는 것이고

$L_T = D_{KL}(q(x_T|x_0)||p(x_T))$ = const

$X_T$가 얼마나 standard Gaussian인지, 그런데 우리는 $\beta_t$를 시간에 따른 constant 로 두었으므로 이 term도 constant이고 학습할 때 `무시`해도 된다

$L_{T-1} = \sum_{t>1}D_{KL}(q(x_{t-1}|x_t,x_0)||p_{\theta}(x_{t-1}|x_t))$

denoising step $p_{\theta}(x_{t-1}|x_t)$ 과 approximated denoising step $q(x_{t-1}|x_t,x_0)$간의 차이를 계산함을 볼 수 있음

model이 `noise를 예측`하도록 `Reparam`

1. $\Sigma_{\theta}(x_t,t) = \sigma_t^2I$, $\sigma$는 $\beta$에 관한 time dependent constants
2. $\mu_\theta(x_t,t)$ for $p_{\theta}(x_{t-1}|x_t)$ using  $x_t(x_0,\epsilon)$

⭐⭐⭐

$$
p_\theta(x_{t-1}|x_t) = N(x_{t-1};\mu_{\theta}(x_t,t),\sigma_t^2I)
$$

$$
\mu_{\theta}(x_t,t) = \frac 1 {\sqrt {\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar \alpha_t}}\epsilon_{\theta}(x_t,t))
$$

**sample** $x_{t-1}$ = $\frac 1 {\sqrt {\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar \alpha_t}}\epsilon_{\theta}(x_t,t))$ + $\sigma_t$**z**,      **z** $\sim N(0,I)$

⭐⭐⭐

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b1b82858-adcf-4e8f-8ee0-b4fb748bd636/Untitled.png)

⇒`$\epsilon_{\theta}$ 는 $x_t$와 t를 받고 noise를 예측`한다.

$x_t$는 $x_0$로 부터 sampling이 가능하도록 위에서 reparam했다.

$L_0 = -\log p_{\theta}(x_0|x_1)$

`Reconstruction term`

⭐⭐⭐

**Simplified Loss**

$$
L_{simple}(\theta) = \mathbb E_{t,x_0,\epsilon} [||\epsilon -\epsilon_{\theta}(\sqrt{\bar \alpha_t}x_0  + \sqrt{1-\bar\alpha_t}\epsilon,t) ||^2]
$$

⭐⭐⭐

t = 1 일 때

$L_0$ 즉, $-\log p_{\theta}(x_0|x_1)$를 minimize

t> 1일 때

$L_{t-1}$에서 식 정리하면 나오는 coefficient $\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}$를 버려서 higher noise level(higer t)에 전(coefficient 가 있을 때)보다 더 큰 weight를 주고 small t에 대해선 더 작은 weight를 줘서 더 좋은 sample quality를 얻었다. (small t에선 model이 작은 양의 noise만 denoise하도록 학습을 시키기 때문, 그래서 더 어려운 large t에 집중하도록 만듦)

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9f379455-4bf7-4088-a3fa-49fa0bda4539/Untitled.png)

<aside>
💡 random하게 timesteps t를 뽑고, $x_0$와 t를 이용해 $q(x_t|x_0)$ 로부터 $x_t$를 구함
이 $x_t$와 t를 우리 모델에 넣고 epsilon을 뽑음
이 epsilon과 ($x_0$와 정확히 같은 dimension을 갖는) noise를 뽑고 MSE loss 때리면 됨

</aside>

### Model Architectue

model의 input과 output의 dimension이 같아야한다. 본 논문에선 U-Net을 사용했다. U-Net은 Residual Block, self-attention block이 있다.

diffusion의 timestep t가 position embedding을 한 후residual block에 전달되는 식으로 모델에 t가 입력된다.

**U-Net**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/22fdc1ea-d205-47fe-9d1f-a9715d5f4c05/Untitled.png)

$\epsilon_{\theta}$ **model using U-Net**

input `$(x_t, t)$`

output `noise`

제거해야할 noise

**Implementation Code**

```python
TODO
Skeleton code 작성해보기
```
