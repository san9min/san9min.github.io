---
title: "Word to Vector"
date: 2022-07-18
readingTime: 15
thumbnail: /images/word2vec_intro/thumbnail.png
tags: [NLP, Word Embedding, Word2Vec]
category: [Tech Review]
---

### 🤖 GPT-3와 단어의 의미

GPT 같은 언어 모델은 말 그대로 AGI로 가는 첫걸음이다.  
그 시작은 아주 단순한 질문에서 출발한다.

> **"단어의 의미를 컴퓨터가 어떻게 알 수 있을까?"**

사전적으로 의미는 이런 것들이다:

- 단어나 문장이 표현하는 개념
- 사람이 단어를 통해 전하려는 생각
- 글이나 예술작품 안에서 드러나는 메시지

언어학에서는 보통 `기호(signifier)`와 `지시 대상(signified)`의 관계로 설명한다.  

이건 **지시적 의미론(denotational semantics)**이라고 부른다.


---

### 💻 의미를 컴퓨터에 어떻게 표현할까?

초기 자연어처리(NLP)에서는 WordNet 같은 자원을 사용했다.  
WordNet은 유의어나 상위어 관계를 정리한 대형 사전이다.

하지만 이런 자원에는 여러 한계가 있다:

- 의미의 **미묘한 차이**를 반영하지 못한다  
  *(예: “proficient”는 항상 “good”과 같지 않다)*
- 새로운 단어나 **은어(slang)** 를 반영하기 어렵다  
  *(예: wicked, ninja, bombest 등)*
- 사람이 계속 업데이트해야 하므로 **효율이 낮고 주관적이다**
- **단어 간 유사도**를 수치로 계산하기 어렵다

---

### 🧾 기존 방식: 단어를 기호로 표현하기

과거 NLP에서는 단어를 **고유한 기호**로 다뤘다.  
예: `hotel`, `conference`, `model`

이 방식에서는 단어를 **one-hot vector**로 표현한다:

- `motel = [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0]`
- `hotel = [0 0 0 0 0 0 0 0 1 0 0 0 0 0 0]`

#### 문제 1

벡터의 차원이 **어휘 수만큼** 커지고,  
단어 간 관계나 유사성을 전혀 알 수 없다.

예를 들어, 사용자가 "Seattle motel"을 검색했을 때  
"Seattle hotel" 관련 문서를 보여주고 싶어도  
두 단어의 벡터가 **서로 직교(orthogonal)**하기 때문에  
관련 문서를 찾기 어렵다.

#### 문제 2

one-hot vector는 **자연스러운 유사도 개념이 없다.**

---

### 📍 해결책: 문맥으로 단어를 표현하기

#### 분포 의미론 (Distributional Semantics)

> “You shall know a word by the company it keeps.”  
> — 단어는 주변 단어들을 통해 의미를 갖는다

즉, 어떤 단어가 자주 함께 등장하는 **문맥(context)**을 보면  
그 단어의 의미를 파악할 수 있다.

예를 들어 `banking`이라는 단어가 들어간 문장은 다음과 같다:

- "... government debt problems turning into `banking` crises ..."
- "... unified `banking` regulation ..."
- "... India gave its `banking` system a boost ..."

이 주변 단어들이 `banking`의 의미를 구성하는 정보가 된다.



### 🧱 Word Vectors: 단어를 벡터로 표현하기

이제 단어를 실수(real number)로 이루어진 **밀집 벡터(dense vector)**로 표현한다.  
이를 **워드 임베딩(word embedding)** 또는 **신경망 단어 표현(neural word representation)**이라고 한다.

같은 문맥에서 자주 등장하는 단어들은 **비슷한 벡터**를 갖도록 훈련된다.

예:


```
banking = [0.286, 0.792, -0.177, ..., 0.271]
```




### ⚙️ Word2Vec: 벡터를 학습하는 알고리즘

**Word2Vec**은 대규모 텍스트로부터 단어 벡터를 학습하는 알고리즘이다.

#### 핵심 아이디어

- 텍스트 전체를 훑으면서, 각 위치마다 중심 단어(center word)와 주변 단어(context word)를 찾는다.
- 중심 단어가 주어졌을 때, 주변 단어가 나올 **확률**을 계산한다.
- 이 확률이 높아지도록 벡터들을 계속 업데이트한다.



### 🎯 목표 함수(Objective Function)

Word2Vec의 학습 목표는  
**주어진 중심 단어로 주변 단어들을 잘 예측할 수 있도록** 벡터를 조정하는 것이다.

목표 함수는 다음과 같다:

$$
J(\theta) = - \frac{1}{T} \sum_{t=1}^T \sum_{-m \le j \le m, j \ne 0} \log P(w_{t+j} | w_t; \theta)
$$



### 📈 확률 계산 방법

확률은 소프트맥스(softmax)를 사용해 계산한다:

$$
P(o|c) = \frac{e^{u_o^T v_c}}{\sum_{w \in V} e^{u_w^T v_c}}
$$

- $v_c$: 중심 단어의 벡터
- $u_o$: 주변 단어의 벡터



### 🔧 벡터 업데이트: 관측값 vs 기대값

목표 함수를 중심 단어 벡터 $v_c$로 미분하면

$$
\frac{\partial \log P(o|c)}{\partial v_c} = u_o - \sum_{x \in V} P(x|c) u_x
$$

- 첫 번째 항: 실제로 관측된 주변 단어의 벡터
- 두 번째 항: 모델이 기대하는 주변 단어 벡터들의 평균

**두 값이 비슷해지도록 벡터를 업데이트**하는 것이 Word2Vec의 핵심이다.

<aside>
단어 간 유사도를 벡터 내적이나 cosine similiarity로 계산할 수 있고
의미를 벡터 공간 위의 방향과 거리로 표현할 수 있다
</aside>

