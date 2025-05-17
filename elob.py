import numpy as np
import matplotlib.pyplot as plt

# ① 1-D 가우시안 예시 ─ posterior vs variational
z = np.linspace(-5, 5, 400)
p = 1/np.sqrt(2*np.pi*1.3**2) * np.exp(-(z-1)**2   /(2*1.3**2))   # p(z|x)
q = 1/np.sqrt(2*np.pi*0.8**2) * np.exp(-(z-1.8)**2 /(2*0.8**2))   # q(z)

plt.figure(figsize=(9,4))

# ② KL 부위 살짝 채우기(시각적 강조)
plt.fill_between(z, np.minimum(p,q), color="#f7f7f7", zorder=1)

plt.plot(z, p, lw=3, label=r'$p(z\,|\,x)$ (true posterior)',  color="#ff8833")
plt.plot(z, q, lw=3, label=r'$q_\phi(z)$ (variational)',      color="#7c6bf3")

# ③ KL 화살표
plt.annotate(r'$\mathrm{KL}(q\;\|\;p)$',
             xy=(2.5, 0.05), xytext=(3.2, 0.12),
             arrowprops=dict(arrowstyle='-[,widthB=4', lw=1.4, color='#555'),
             fontsize=12)

# ④ ELBO 식
plt.text(-4.8, 0.14,
         r'$\log p_\theta(x) \;=\; \underbrace{\mathcal{L}_{\text{ELBO}}}_{\text{evidence lower bound}} \;+\; \mathrm{KL}(q_\phi \,\|\, p_\theta)$',
         fontsize=13)

plt.xlabel(r'$z$'); plt.ylabel('density')
plt.xlim(-5,5); plt.ylim(0,0.5)
plt.legend(frameon=False, loc='upper right')
plt.tight_layout()
plt.show()
