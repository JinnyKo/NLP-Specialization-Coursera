## 마르코프 체인 (Markov Chains) 
> 미래의 상태가 오직 현재 상태에만 의존하고 과거 상태와는 무관한 확률 과정을 설명하는 수학적 모델..
> 간단히 말해서 **마르코프 체인에서는 다음 상태로의 전이 확률이 현재 상태에만 의존하며, 이전의 경로나 상태에는 영향을 받지 않는다**고 할 수있다.
> 이 개념은 자연어 처리에서 뿐만 아니라 경제학, 게임이론 등에서도 자주 쓰인다.
> 자연어 처리에서는 단어 또는 문자의 시퀀스를 생성할 때 마르코프 체인을 사용할 수 있는데, 단어나 문자의 연쇄가 마르코프 프로세스를 따른다.

```
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# nltk 리소스 다운로드
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# 샘플 문장
sentence = "Markov chains can be used for various applications in natural language processing."

# 문장을 단어로 토큰화하고 각 단어에 대해 품사 태깅
tokens = word_tokenize(sentence)
tagged_tokens = pos_tag(tokens)

print(tagged_tokens)

```

>거대모델을 학습시키는걸 기반으로 생각하니 계속 의문점이 들었다. GPT와 같은 거대모델을 학습시킬 때는 마르코프 체인을 따르지 않는다는 점.
> - **마르코프 체인** : 간단한 예측, 텍스트 생성, 날씨 변화 모델링 등 제한된 정보를 바탕으로 하는 예측에 주로 사용됨.
> - **트랜스포머** : 고급 자연어 처리 작업, 기계 번역, 텍스트 요약, 질문 응답 시스템 등 복잡한 언어 모델링과 관련된 작업에 사용됨.


## 마르코프체인은 그럼 언제 쓰나
비록 마르코프체인이 최신 딥러닝 기술에 비해 단순 할 수 있지만 여전히 다양한 분야에서 쓰이고 있다. 
1. **텍스트 생성**:
 간단한 텍스트나 음악, 리듬 생성. 예를 들어, 간단한 마르코프 체인을 사용하여 문자열을 생성하거나, 간단한 시나 가사를 만드는 데 활용할 수 있음.

2. **상태 예측 모델링**:
경제학, 기상학, 게임 이론, 공학 응용 분야에서 미래 상태 예측하는 데 마르. 예를 들어, 날씨의 변화, 주식 시장의 변동성, 제품의 수명 주기 등을 모델링하는 데 유용.

3. **경로 추정**:
 특정 상태에서 다른 상태로 이동하는 경로나 패턴 예측. 예를 들어, 웹사이트 사용자의 탐색 경로, 도시 내 교통 흐름 분석 등

4. **생물학과 유전학**:
 DNA, RNA, 단백질의 서열 모델링하고 예측.

## 마르코프체인의 한계 

기본 마르코프 모델은 모델의 간결함과 무기억성(memorylessness) 때문에 한계가 있다. 마르코프 모델에서 미래 상태(예: 다음 단어)의 예측은 오직 현재 상태(혹은 고정된 수의 이전 상태들)에만 의존하고, 그 이전의 상태나 문맥은 고려하지 않는다. 

**문맥의 제한성**: 기본 마르코프 모델은 고정된 수의 이전 상태만을 고려한다. 이는 모델이 더 넓은 문맥을 파악하는 데 한계가 있음을 의미하며, 자연어의 경우 **특히 문맥이 중요하기 때문에 이러한 제한은 큰 단점이 될 수 있다.**

**품사와 단어의 관계 무시**: 기본 마르코프 모델은 단어의 연쇄만을 모델링하거나 품사의 연쇄만을 모델링할 수 있으며, 둘 사이의 관계를 동시에 모델링하지 못한다. 품사가 단어 선택에 중요한 역할을 하는 언어의 특성을 고려할 때, 이는 큰 제약임.

 > ## 그래서 나온게 Hidden Markov Model(HMM)

## Hidden Markov Model(HMM)

Hidden Markov Model (HMM)은 마르코프 체인을 확장한 모델로, 관찰할 수 없는 숨겨진 상태(hidden states)가 시퀀스 데이터를 생성하는 과정을 모델링 한다.
HMM은 주로 두 가지 기본 요소인 전이 행렬(Transition Matrix)과 방출 행렬(Emission Matrix)을 사용.

- 전이 행렬 (Transition Matrix):
전이 행렬은 숨겨진 상태 간의 전이 확률. 즉, 한 상태에서 다른 상태로 이동할 확률을 정의한다. 이 행렬의 각 요소는 특정 숨겨진 상태에서 다음 시간 단계의 다른 숨겨진 상태로 전이할 확률을 나타낸다.

- 방출 행렬 (Emission Matrix):
방출 행렬은 각 숨겨진 상태에서 관찰 가능한 각 출력 심볼(또는 상태)이 관찰될 확률을 나타낸다. 이 행렬은 숨겨진 상태에 기반해 관찰된 상태가 나타날 확률을 제공함
==> 무슨 말이냐 하면, 인간을 단어를 보고 품사를 함께 생각 낼 수 있지만, 훈련하는 모델은 그렇지 않기 때문에 Emission Matrix를 사용해 단어와 품사간의 관계성을 제공해 줘야함.

> - 숨겨진 상태: 각 단어의 품사 (예: 명사, 동사, 형용사 등)
> - 관찰 상태: 문장에서의 실제 단어
> - **어떤 품사가 주어졌을 때 다음 품사가 무엇인지 (Transition Matrix))**
> - **특정 품사일 때 어떤 단어가 나타날 확률 (Emission Matrix)**
>   
> 이렇게 모델을 설정하면 전이 행렬을 사용하여 **특정 품사 다음에 어떤 품사가 나타날 확률을 모델링하고**, 방출 행렬을 사용하여 **각 품사에 대해 특정 단어가 나타날 확률**을 모델링할 수 있다.

### 예를들어 
HMM에서는 "I"가 명사(Noun)인 상황에서 다음 단어 "love"가 동사(Verb)로 사용될 확률을 전이 행렬을 통해,
그리고 "love"가 Verb일 때 실제로 'love'라는 단어가 관찰될 확률을 방출 행렬을 통해 계산한다. 

```
from hmmlearn import hmm
import numpy as np

# 품사는 숨겨진 상태로, 단어는 관찰 가능한 상태로 설정
states = ["Noun", "Verb"]
n_states = len(states)

observations = ["I", "read", "a", "book"]
n_observations = len(observations)

# 시작 확률 (각 품사가 문장의 시작일 확률)
start_probability = np.array([0.5, 0.5])

# 전이 확률 행렬 (품사에서 품사로의 전이 확률)
transition_probability = np.array([
  [0.7, 0.3],  # Noun -> Noun, Noun -> Verb
  [0.4, 0.6],  # Verb -> Noun, Verb -> Verb
])

# 방출 확률 행렬 (품사에서 각 단어가 나올 확률)
emission_probability = np.array([
  [0.1, 0.3, 0.4, 0.2],  # Noun -> I, read, a, book
  [0.3, 0.3, 0.2, 0.2],  # Verb -> I, read, a, book
])

# HMM 모델 생성 및 파라미터 설정
model = hmm.MultinomialHMM(n_components=n_states)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

# 관찰된 단어 시퀀스를 통한 품사 예측
# 예를 들어, "I read a book" 문장에 대한 품사 예측
observed_sequence = np.array([[0, 1, 2, 3]]).T
logprob, sequence = model.decode(observed_sequence, algorithm="viterbi")

print("Predicted states:")
for i in sequence:
    print(states[i])
```
## Transition Matrix
![image](https://github.com/JinnyKo/NLP-Specialization-Coursera/assets/93627969/1735f6c0-8e02-4caa-b10f-b185aabc7cbb)

### 때때로 두 개의 POS 태그가 서로 앞에 표시되지 않는 경우 => Smoothing 
예를 들어, 전이 확률 행렬에서 특정 품사 A에서 품사 B로의 전이가 학습 데이터에 전혀 나타나지 않았다면, 해당 확률은 0으로 설정된다. 
그러나 스무딩을 적용하면 이 확률에 아주 작은 값을 부여하여 완전히 0이 되지 않도록 조정할 수 있다. 이렇게 하면 학습 데이터에는 나타나지 않았지만 실제 사용 상황에서 발생할 수 있는 전이를 모델이 완전히 무시하지 않게 된다. (데이터 하나하나 소중...) 

엡실론(ϵ)은 스무딩에서 사용하는 매우 작은 양수 값이 값은 . 특정한 고정값을 가지기보다는 문맥에 따라 그 크기가 조정될 수 있다. 
스무딩에서 엡실론 값은 보통 실험적으로 결정되며, 모델의 성능에 미치는 영향을 고려하여 최적화된다. 
일반적으로는 매우 작은 값(예: 0.01, 0.001, 1e-5 등)을 사용하여, 학습 데이터에 없는 시퀀스에 대해 0이 아닌 매우 작은 확률을 부여하고, 
동시에 학습 데이터에 있는 시퀀스의 확률 분포를 크게 왜곡하지 않도록 한다. 

## Emission Matrix
![image](https://github.com/JinnyKo/NLP-Specialization-Coursera/assets/93627969/40f0bee6-be70-4c24-b8b3-06d1f87bcaab)

















