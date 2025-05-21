---
title: "Diffusion : Kick-off"
date: 2023-01-15
readingTime: 15 
thumbnail: /images/diffusion_kickoff/thumbnail.png
tags: [Generative AI, Diffusion]
category : [Tech Review]
---
### Genetrative Model Framework
Genetraion은 크게 두가지 framework으로 나뉜다.
- **Likelihood-based**
    - Autoregressive Models
    - Variational Autoencoders
    - Flow-based Models
    - **`Diffusion models`**
- **Implicit model**
    - Generative Adversarial Networks(GAN)

![B948109D-1EBD-4BF6-B739-FB8EF24AE95E.png](/images/diffusion_kickoff/01.png)

diffusion model은 likelihood based이다.

---

### 📚 Background  

Diffusion은 확률 기반의 process이다. 먼저 핵심 확률 개념 3가지를 정리하자.

#### (1) KL-Divergence

> `두 확률 분포가 얼마나 다른지`를 계산, minimize를 통해 근사


$$
D_{KL}(q\||p)=
\begin{cases}
-\displaystyle\sum_i q_i \log\frac{p_i}{q_i}, & \text{(discrete form)} \\\\
-\displaystyle\int q(x)\log\frac{p(x)}{q(x)}, & \text{(continuous form)}
\end{cases}
$$

식을 전개하면 Self Entropy term과 Cross Entropy으로 분리할 수 있다.

$$
D_{KL} = -H(q) + H(q,p)
$$

예를 들어, continuous form에 대해

$$
\int q(x) log(q(x))dx + \int q(x)(-log(p(x))dx\\ = -H(q) + H(q,p)
$$

로 쓸 수 있다.


식을 자세히 보자.
우리가 q를 p로 근사시키기 위해, KL divergence를 minimize한다고 했을 때
Self Entropy term은 q의 variance를 증가시키켜 넓게 퍼진 분포가 되려는 경향을 갖게하고 
Cross Entropy term은 p 분포에서 likelihood가 가장 높은 지점에서 Delta function처럼 되게하려는 경향을 갖게 할 것이다.  

이 두 term을 통해 (싸우는 느낌?) q가 p로 근사가 된다.

**KL divergence의 특성** 

* **항상 0 이상**이다. CE는 아무리 낮아져봤자 (즉, q와 p가 같은 분포가 된다 했을 때) self-entropy이다. 그러므로 최솟값이 0이고, 절대 음수가 될 수 없다. 

* 엄밀히는 거리 개념이 아니다. 

* 일반적으로 $D_{KL}(p|q) \neq D_{KL}(q|p)$이다.

#### (2) Bayes Rule  
> 복잡한 posterior p(z∣x)를 prior·likelihood·evidence 항으로 분해, ELBO와 KL 식 도출

$$
P(H \mid E) \=\ \frac{P(H)\,P(E \mid H)}{P(E)}
$$

*E : Evidence(sample x, 증거·관측 데이터), H : Hypothesis(latent z ,가설)*

| 기호            | 용어                              |의미                                                      |
| ------------- | ---------------------------------- | ----------------------------------------------------------- |
| $P(H)$        | **Prior probability**              | 관측 전에 가설 $H$가 참일 사전확률                                       |
| $P(E \mid H)$ | **Likelihood**                     | $H$가 참일 때 증거 $E$가 나타날 가능도 <br>→ 가설 $H$가 증거 $E$를 얼마나 잘 설명하는지 |
| $P(E)$        | **Evidence / Marginal likelihood** | 가설을 구분하지 않고 $E$가 관측될 전체 확률                                  |
| $P(H \mid E)$ | **Posterior probability**          | 증거 $E$를 본 뒤 가설 $H$가 참일 사후확률                                 |


#### (3) Monte Carlo Method
> 적분 대신 **샘플 평균**으로 근사

랜덤 표본을 뽑아 (`sampling`을 통해) 함수값을 확률적으로 계산하는 방식으로, 결국 근사(`approximation`) 이야기다.


예를들면

$$
\int p(x)f(x)dx = E_{x \sim p(x)}[f(x)] \approx \frac 1 K \sum_i^K f(x_i), x_i \sim p(x)
$$

확률 밀도함수 p(x)를 따르는 x에대한 f(x)의 기댓값을 구하고 싶다했을 때 p(x)에서 K개의 샘플을 뽑아 이로 계산해도 괜찮다는 이야기이다.

---


### 🤔 왜 Variational Inference(VI)가 필요할까?  
> 직접 적분이 불가능한 posterior $p_\theta(z\mid x)$ 대신, tractable 근사분포 $q_\phi(z\mid x)$로 문제를 풀어 $\log p_\theta(x)$를 최적화하기 위해  


**먼저, Likelihood-based 모델이 최적화하려는 핵심 값 두 가지**

1. **Likelihood**  
   $$
   \log p_\theta(x)
   $$
   → 모델이 관측 $x$ 를 얼마나 잘 설명하는지의 척도

2. **Posterior**  
   $$
   p_\theta(z \mid x)
   $$
   → 주어진 $x$ 에서 잠재 변수 $z$ 가 취할 분포

Bayes Rule로 posterior $p_\theta(z\mid x)$를 정의할 수는 있지만  **정규화 상수** $p_\theta(x)$ 계산이 어렵고 **고차원 적분** 때문에 실제 값을 구하기가 사실상 불가능하다.  


$$
p_\theta(z\mid x) = \frac{p_\theta(x\mid z) p_\theta(z)}{p_\theta(x)}
\quad\text{where}\quad p_\theta(x)=\int p_\theta(x\mid z)p_\theta(z) dz
$$


따라서 **Variational Inference**는 다루기 쉬운 분포 $q_\phi(z\mid x)$로 $p_\theta(z\mid x)$를 **근사**하고, 두 분포의 차이를 **KL Divergence**로 최소화한다.  


이 과정의 최적화 목표가 **ELBO(Evidence Lower Bound)** 이다.


---


### 📐 ELBO : Evidence Lower Bound  


정리하면, Variational Inference의 궁극적 목적은 복잡한 posterior $p(z\mid x)\$ 를 다루기 쉬운 $q_\phi(z\mid x)\$ 로 근사하는 것.  
여기서 latent variable z 의 사전확률 분포 p(z)는 x와 무관해 가장 간단하고 예쁜 Gaussian이라 하자. 그리고 q를 $\phi$, p를 $\theta$로 parameterize하자. 


$$q_\phi^{*} = \underset{q_\phi \in \mathcal Q}{\arg\min} D_{KL}\(q_\phi(z \mid x) || p_\theta(z \mid x))
$$

#### (1) KL 분해
Bayes Rule에 의해 우리는 posterior p(z|x)를 p(z), p(x), p(x|z)로 쓸 수 있다.
그러므로 Bayes Rule을 이용해 KL divergence를 표현하면


$$
\begin{aligned}
D_{KL}\\bigl(q_\phi(z\mid x)\|p_\theta(z\mid x)\bigr)
  &= \int q_\phi(z\mid x)\
       \log\frac{q_\phi(z\mid x)}{p_\theta(z\mid x)}\
       \mathrm dz \\\\[6pt]
  &= \int q_\phi(z\mid x)\
       \log\frac{q_\phi(z\mid x)p_\theta(x)}
                {p_\theta(x\mid z)p_\theta(z)}\
       \mathrm dz \\\\[6pt]
  &= \int q_\phi(z\mid x)\
       \log\frac{q_\phi(z\mid x)}{p_\theta(z)}\
       \mathrm dz
     + \log p_\theta(x)
     - \int q_\phi(z\mid x)\
         \log p_\theta(x\mid z)\
         \mathrm dz \\\\[6pt]
  &= D_{KL}\bigl(q_\phi(z\mid x)\|p_\theta(z)\bigr)
     + \log p_\theta(x)
     - \mathbb E_{z \sim q_\phi(z\mid x)}
       \bigl[\log p_\theta(x\mid z)\bigr]
\end{aligned}
$$


로 정리된다. 


그런데 여기서 **$\log p_{\theta}(x)$는 intractable하다. 그래서 우리는 tractable한 lower bound(ELBO)를 잡고 이를 maximize하는 방식을 취한다.**


Expectation : $D_{KL} (q_{\phi}(z|x) || p(z|x))$를 minimize하는 $\phi$를 찾자

Maximization :  $\phi$를 고정하고 $\log p_{\theta}(x)$의 lower bound를 maximize하는 $\theta$를 찾자

<aside>
<bold>log p(x) 를 evidence, likelihood라고 한다</bold>

 $\theta$로 parameterized된 우리의 model이 observed data x에 대해 marginal probability를 계산했을 때 만약 우리 모델이 잘 학습이 되었다면 높은 값을 내놓을 것이다. 즉, 학습 중에 $\theta$를 잠시 fix해놓고 evaluation을 했을 때 높은 값을 내놓고 있다면 우리는 잘 가고 있다는 것이다. 그래서 $\log p(x;\theta)$를 우리가 잘 가고 있다는 의미에서  evidence라 한다. 
</aside>


$D_{KL} (q_{\phi}(z|x) || p(z|x))\ge 0$ 이므로 

$$
\log p_{\theta}(x) \ge E_{z \sim q(z)}[\log p_{\theta}(x|z)] - D_{KL}(q_{\phi}(z|x)||p(z)) 
$$

가 된다.

#### (2) ELBO 정의

$$
{\text{ELBO}} = E_{z \sim q_{\phi}(z|x)}[\log p_{\theta}(x|z)] - D_{KL}(q_{\phi}(z|x)||p(z)) 
$$


$E_{z \sim q(z|x)}[\log p_{\theta}(x|z)]$

#### 1st term : Reconstruction Error
* generative model(Decoder in VAE)
* Decoder가 데이터를 얼마나 잘 복원하는가



$D_{KL}(q(z|x)||p(z)) \text{ or } E_{q(z|x)} [\log \frac {q(z|x) }{p(z)}]$

#### 2nd term : Regularization term`
* inference model(Encoder in VAE)
* $q(z\mid x)$ 가 prior $p(z)$ 와 얼마나 비슷한가


를 나타댄다.

$$
\log p_{\theta}(x) \ge {\text{ELBO}}
$$
이므로 ELBO를 최대화하면 곧 KL도 줄이면서 $\log p_\theta(x)$를 끌어올리는 효과를 얻는다.


---

### 🏁 나가며

#### ① ELBO가 하는 일  
* **ELBO**는 직접 계산이 어려운 **likelihood $p_\theta(x)$** 의 *안전한 하한(lower bound)*.  
* 이 하한을 **최대화**하면 실제 $\log p_\theta(x)$ 도 함께 끌어올릴 수 있다.  


#### ② Diffusion·VAE 등 모델의 학습 법칙  
* **Diffusion** 모델은 likelihood-based 계열 → 학습 단계에서 **ELBO 최대화**로 파라미터 최적화  
* VAE·Flow 등 모든 **Variational 모델**도 동일한 목표를 따른다.  


#### ③ Variational Inference 관점  
* 복잡한 사후분포 $p_\theta(z\mid x)$ 대신 **근사 분포 $q_\phi(z\mid x)$** 사용  
* 두 분포의 차이를 **KL Divergence**로 측정·최소화  
* 결론적으로 나온 ELBO는 *Reconstruction Error* 와 *Regularization term* 으로 구성 

> *Training* 단계에서는 **ELBO** 를 최대화해 $\log p_\theta(x)$ 를 간접적으로 높인다.
> *Inference* 단계에서는 $q_\phi(z\mid x)$ 로 posterior를 근사한다.  


