---
title: "Diffusion : DDPM"
date: 2023-01-22
readingTime: 20 
thumbnail: /images/diffusion_ddpm/thumbnail.jpg
tags: [Generative AI, Diffusion, DDPM]
category : [Tech Review]
---

## Denoising Diffusion Probabilistic Models

[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

- parameterized Markov chain
- trained using variational inference
- <marck>`learn to reverse a diffusion process` </mark>

![Untitled](/images/diffusion_ddpm/00.png)

Diffusion model은 latent variable 모델이다.
*latent : a hidden continuous feature space*

**GOAL**
<figure class="eq">
$$
p_{\theta} (x_0) = \int p_{\theta}(x_{0:T})dx_{1:T}
$$
</figure>
$x_1,..x_T$ 는 latent들이고 data $x_0 \sim q(x_0)$와 같은 dimension을 갖는다.

model은 이 reverse process를 학습하고 새로운 data를 generation한다.

---

##  >> Forward Process (diffusion process)

> `Gaussian noise를 더해가는 과정`


### (1) Markov chain - Noising process

$$
q(x_{1:T}|x_0) = \Pi_{t=1}^Tq(x_t|x_{t-1}) 
$$


forward 과정은 Markov chain으로 formulate할 수 있다.  
Markov chain은 각 step은 오직 직전 step에만 의존함을 의미한다.  
여기서 $q(x_{1:T})$는 q를 timestep 1부터 T까지 반복해서 노이즈를 가함을 의미하는 notation이다.


$$q(x_{t}|x_{t-1}) = N(x_t ; \sqrt {1-\beta_t}x_{t-1},\beta_tI)$$

각 스텝에서 Gaussian noise를 더한다. 한 스텝 전이할 때마다,평균을 스케일링하고, 분산만큼 가우시안 노이즈를 추가한다.  

**$\beta_t$ (분산 스케줄)**  

여기서 $\beta_t$는 variance schedule이고 I가 identity이므로 각 dimension은 같은 std를 갖는다. 
상수로 둬도 되고, 시간에 따른 변수로 두어도 된다. 
논문에선 higer t로 갈수록 커지는 방향으로 linear하게 두었는데 다른 논문에선 cosine shcedule 이 잘 됐다고 한다.

$\beta_t$를 이용해 scaling한 후 더해주는 이유는 variance가 diverge하는 것을 막기 위함으로 볼 수 있다. 
Gaussian noise를 더해가면, 최종 step (time T)에서는 표준 정규분포에 $N(x_T;0,I)$ 가깝게 수렴한다. 
이 단순한 prior 덕분에 Reverse Process 는 “완전한 노이즈 → 원본” 만 학습하면 된다

<aside>
$\beta _t$를 너무 크게 잡으면 분산이 폭주하고, 너무 작으면 느리게 파괴되므로 스케줄 디자인이 핵심이다.
</aside>

그런데 여기서 어떤 순간 t ( $0 \le t \le T)$에서 $x_t$를 알고 싶다고한다면, 위의 식을 이용해 반복적인 계산을 수행하면 된다. 그러나 t 가 크다면 이는 좋은 방법이 아닐 것이다.

### (2) Reparameterization trick - 한번에 $x_t$ 를 sampling

만약 우리가
$\alpha_t = 1-\beta_t , \bar \alpha_t = \Pi_{s=0}^t \alpha_s$ 라고 잡는다면, t에서 $x_t$를 sampling하는 것을 closed form으로 쓸 수 있을 것이다.

<figure class="eq">

$$
q(x_t|x_0) = N(x_t;\sqrt {\bar \alpha_t} x_0, (1- {\bar \alpha_t})I)
$$



$$
\text{sample} =\mu +\sqrt{\sigma^{2}}\;\epsilon,\qquad\epsilon \sim \mathcal N(0,1)
$$

</figure>
    
**Proof**
    $$\text{let  } \epsilon_0, \cdots \epsilon_{t-2},\epsilon_{t-1} \sim N(0,I)$$
    $$x_ t = \sqrt{(1-\beta_t)}x_{t-1} + \sqrt {\beta_t} \epsilon_{t-1} \\\\ =\cdots  \\\\=\sqrt{\bar \alpha_t}x_0 + \sqrt{1-\bar{\alpha_t}}\epsilon_0$$

증명의 핵심 논리는 다른 variance를 갖는 두개($\sigma_1^2,\sigma_2^2$)의 Gaussians을 merge해 새로운 distribution(with variance $\sigma_1^2+\sigma_2^2$ )을 만드는 것이다.


확인을 해보면 t → $\infty$로 갈때 $q(x_t|x_0)$가 $N(x_t;0,I)$로 감도 볼 수 있다

이제 우리는 any timestep t에서 noise를 sampling할 수 있게 됐다.
이를 통해 $x_t$를 $x_0$와 $\epsilon$의 함수로 볼수 있게 된다.

$$
x_t = \sqrt{\bar \alpha_t}x_0 + \sqrt{1-\bar{\alpha_t}}\epsilon_0
$$

<span>or</span>

$$
x_0 = \frac{1}{\sqrt{\bar \alpha_t}}(x_t - \sqrt{1-\bar \alpha_t}\epsilon)
$$

<aside>
모든 중간 스텝을 거치지 않아도, $x_0$ 과 표준정규 노이즈 $\epsilon$ 만으로 즉시 $x_t$ 생성 가능하다.
이 트릭 덕분에 임의 시점의 노이즈 상태를 빠르고 안정적으로 샘플링할 수 있고, 미분 가능한 parameter로 업데이트에 사용할 수 있다.
</aside>


---

## << Reverse Process (denoising process)

> `모델이 학습할 과정`

$q(x_{t-1}|x_{t})$를 원하나 어려워서 neural network를 이용

forward process의 Gaussian noise가 충분히작을 때 reverse process 또한 Gaussian이 되고 이는 neural network를 이용해 근사시켜 mean과 variance를 예측하는 모델을 만들 수 있다.

$$
p_{\theta}(x_{t-1}|x_t) = N(x_{t-1};\mu_{\theta}(x_t,t), \Sigma_{\theta}(x_t,t))
$$

Reverse process는 $p(x_T) = N(x_T;0,I)$부터 출발해 (learned) Gaussian trainsition을 하는 Markov chain이다. 

즉, trajectory는

$$
p_{\theta}(x_{0;T}) = p(x_T)\Pi_{t=1}^T p_{\theta}(x_{t-1}|x_t)
$$

로 fomulate할 수 있다.

neural network에 timestep t를 conditioning하면 model은 각 time step의 Gaussian의 mean과 variance를 예측할 수 있게된다.

---

## 🏋️ Training

> `ELBO` on the negative log likelihood를 optimize

$$
E[-\log p_{\theta}(x_0)] \le E_q[-\log \frac {p_\theta (x_{0:T})}{q(x_{1:T}|x_0)}] \\ = L
$$

이고 위 식을 정리하면

$$
L = E_q[D_{KL}(q(x_T|x_0)||p(x_T))+\sum_{t>1}D_{KL}(q(x_{t-1}|x_t,x_0)||p_{\theta}(x_{t-1}|x_t))-\log p_{\theta}(x_0|x_1)]
$$

이 된다.

<aside>
$q(x_{t-1}|x_t)$는 intractable하지만 $x_0$의 conditioning을 주면 tractable하다고 함 → generative model이 reverse diffusion step으로 generation을 하기 위해선 reference image $x_0$가 필요하다

</aside>

결국 하고 싶은 것은 $p_{\theta}(x_{t-1}|x_t)$와 forward process posteriors $q(x_{t-1}|x_t,x_0)$를 비교하는 것이고

$L_T = D_{KL}(q(x_T|x_0)||p(x_T))$ = const

$X_T$가 얼마나 standard Gaussian인지, 그런데 우리는 $\beta_t$를 시간에 따른 constant 로 두었으므로 이 term도 constant이고 학습할 때 `무시`해도 된다

$L_{T-1} = \sum_{t>1}D_{KL}(q(x_{t-1}|x_t,x_0)||p_{\theta}(x_{t-1}|x_t))$

denoising step $p_{\theta}(x_{t-1}|x_t)$ 과 approximated denoising step $q(x_{t-1}|x_t,x_0)$간의 차이를 계산함을 볼 수 있음

<mark>
model이 `noise를 예측`하도록 `Reparam`
</mark>

1. $\Sigma_{\theta}(x_t,t) = \sigma_t^2I$, $\sigma$는 $\beta$에 관한 time dependent constants
2. $\mu_\theta(x_t,t)$ for $p_{\theta}(x_{t-1}|x_t)$ using  $x_t(x_0,\epsilon)$

<figure class="eq">

$$
p_\theta(x_{t-1}|x_t) = N(x_{t-1};\mu_{\theta}(x_t,t),\sigma_t^2I)
$$

$$
\mu_{\theta}(x_t,t) = \frac 1 {\sqrt {\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar \alpha_t}}\epsilon_{\theta}(x_t,t))
$$

 $sample \\ x_{t-1}$ = $\frac 1 {\sqrt {\alpha_t}}(x_t-\frac{\beta_t}{\sqrt{1-\bar \alpha_t}}\epsilon_{\theta}(x_t,t))$ + $\sigma_t$**z**,      **z** $\sim N(0,I)$

</figure>

$$
\mathbb{E}_{x_0,\epsilon}\left[\frac{\beta_t^{2}}{2\sigma_t^{2}\alpha_t(1-\bar\alpha_t)}\bigl\lVert\epsilon - \epsilon _\theta\bigl(\sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon,t \bigr) \bigr\rVert^{2}\right]
$$


⇒$\epsilon_{\theta}$ 는 $x_t$와 t를 받고 noise를 예측한다.

$x_t$는 $x_0$로 부터 sampling이 가능하도록 위에서 reparam했다.

Reconstruction term
$L_0 = -\log p_{\theta}(x_0|x_1)$



**Simplified Loss**
<figure class="eq">
$$
L_{simple}(\theta) = \mathbb E_{t,x_0,\epsilon} [||\epsilon -\epsilon_{\theta}(\sqrt{\bar \alpha_t}x_0  + \sqrt{1-\bar\alpha_t}\epsilon,t) ||^2]
$$
</figure>

t = 1 일 때

$L_0$ 즉, $-\log p_{\theta}(x_0|x_1)$를 minimize

t> 1일 때

$L_{t-1}$에서 식 정리하면 나오는 coefficient $\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}$를 버려서 higher noise level(higer t)에 전(coefficient 가 있을 때)보다 더 큰 weight를 주고 small t에 대해선 더 작은 weight를 줘서 더 좋은 sample quality를 얻었다. (small t에선 model이 작은 양의 noise만 denoise하도록 학습을 시키기 때문, 그래서 더 어려운 large t에 집중하도록 만듦)

![Untitled](/images/diffusion_ddpm/01.jpg)

<aside>
random하게 timesteps t를 뽑고, $x_0$와 t를 이용해 $q(x_t|x_0)$ 로부터 $x_t$를 구함
이 $x_t$와 t를 우리 모델에 넣고 epsilon을 뽑음
이 epsilon과 ($x_0$와 정확히 같은 dimension을 갖는) noise를 뽑고 MSE loss 때리면 된다.
</aside>

---

##  🛠️ Model Architectue

model의 input과 output의 dimension이 같아야한다. 본 논문에선 U-Net을 사용했다. 
U-Net은 Residual Block, self-attention block이 있다.

diffusion의 timestep t가 position embedding을 한 후residual block에 전달되는 식으로 모델에 t가 입력된다.

**U-Net**

![Untitled](/images/diffusion_ddpm/02.jpg)

$\epsilon_{\theta}$ **model using U-Net**

* input : $(x_t, t)$
* output :`제거해야할 noise`



**Implementation Code**

```python
class ResBlock(nn.Module):
    def __init__(self, c, t_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(t_emb_dim, c)
        self.block    = nn.Sequential(
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1)
        )

    def forward(self, x, t_emb):
        h = self.block(x) + self.time_mlp(t_emb)[:, :, None, None]
        return h + x                            # residual

class UNet(nn.Module):
    def __init__(self, ch=64, t_emb_dim=256):
        super().__init__()
        self.t_embed = nn.Sequential(
            nn.Linear(1, 128), nn.SiLU(), nn.Linear(128, t_emb_dim)
        )
        self.down1 = ResBlock(ch, t_emb_dim)
        self.down2 = ResBlock(ch*2, t_emb_dim)
        self.up1   = ResBlock(ch*2, t_emb_dim)
        self.out   = nn.Conv2d(ch, 3, 1)

    def forward(self, x, t):
        t_emb = self.t_embed(t)
        h1 = self.down1(x, t_emb)
        h2 = self.down2(nn.functional.avg_pool2d(h1, 2), t_emb)
        h  = nn.functional.interpolate(h2, scale_factor=2)
        h  = self.up1(torch.cat([h, h1], 1), t_emb)
        return self.out(h)

def beta_schedule(T, start=1e-4, end=0.02):
    return torch.linspace(start, end, T)

def q_sample(x0, t, √abar, √1mabar, noise):
    return √abar[t][:, None, None, None] * x0 + √1mabar[t][:, None, None, None] * noise

def loss_step(model, x0, t, √abar, √1mabar):
    noise  = torch.randn_like(x0)
    xt     = q_sample(x0, t, √abar, √1mabar, noise)
    pred_n = model(xt, t.float().unsqueeze(-1))
    return ((noise - pred_n) ** 2).mean()
```
