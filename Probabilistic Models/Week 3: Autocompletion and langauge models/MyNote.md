![image](https://github.com/JinnyKo/NLP-Specialization-Coursera/assets/93627969/d1e0db71-f214-443d-aaef-1a0750bdd090)

# N-gram vs HMM 
전 주에서 공부한 HMM과 직접적인 비교를 통해 N-gram의 개념에 대한 이해를 해보고자 한다. 
일단, HMM의 최종적인 목표는(지난 주 과제를 기준으로) 

1. **특정 품사를 가진 단어 예측**:주어진 문장에서 각 단어에 가장 적합한 품사를 예측. 이 경우, 모델은 각 단어의 은닉 상태(품사)와 그 단어(관측된 상태) 사이의 관계를 학습하여, **다음에 올 단어의 품사를 예측** 한다. => 문장 내에서 단어의 순서와 품사 간의 확률적 관계를 이용하는 것이다.

2. **단어가 특정 품사를 갖는 것을 예측**: 이는 HMM을 통해 주어진 단어가 특정 품사를 가질 확률을 예측. HMM은 각 상태(품사) 전이의 확률과 각 상태에서 특정 단어(관측)가 나타날 확률을 모델링하는데, 이 정보를 통해 **문장 내의 각 단어에 대해 가장 가능성이 높은 품사를 태깅**할 수 있다.

> #### **HMM을 사용한 품사 태깅의 주 목적은 특정 단어 뒤에 어떤 단어가 올지 직접 예측하는 것이 아니다!**.
> 대신, HMM은 주어진 단어 시퀀스에서 각 단어의 품사를 예측하는 데 초점을 맞춘다. 즉, 문장 내 각 단어의 품사를 결정하는 것이 주된 목표인것.

#### 따라서, "특정 단어 뒤에 어떤 단어가 올지"를 예측하려면 다른 종류의 언어 모델(예: N-gram, 신경망 기반 모델 등)을 사용하는 것.

# N-gram
: N-gram은 주어진 시퀀스에서 **N개의 연속적인 항목**(보통은 단어)을 분석하는 모델. N-gram 모델에서는 (N-1)개의 단어가 주어졌을 때 N번째 단어의 등장 확률을 계산한다. 

> **N개의 연속적인 항목이란**, 텍스트 내에서 연속적으로 나타나는 N개의 단어(또는 문자)를 의미한다. 이 개념은 단순히 전체 문서에서 임의로 선택된 N개의 단어가 아니라, 실제로 텍스트 내에서 인접해 있는 N개의 단어 시퀀스를 가리킨다.

> 예를 들어, "The quick brown fox jumps over the lazy dog"이라는 문장이 있다고 할 때:
- 2-gram(바이그램)의 경우, 연속적인 항목의 예는 "the quick", "quick brown", "brown fox", "fox jumps". 각각은 문장 내에서 바로 옆에 있는 단어들로 구성된다. 
- 3-gram(트라이그램)에서는 "the quick brown", "quick brown fox", "brown fox jumps" 이런식으로..
  
> N-gram 모델에서는 이러한 N개의 연속적인 단어를 기반으로 하여 N+1번째 단어의 등장 확률을 예측하는것. 예를 들어, 바이그램 모델에서 "quick brown" 다음에 어떤 단어가 올 확률을 계산한다면, 이전에 나타난 "quick brown"이라는 바이그램에 기반하여 다음에 올 수 있는 단어의 확률 분포를 추정하는 것이다. 

#### 따라서 N-gram 모델에서 **N은 고려하는 문맥의 크기**를 나타내며, 이 문맥을 바탕으로 다음 단어를 예측. => **단어의 순서와 문맥이 모델에 반영**되어 더 정확한 언어 모델링이 가능해진다. 

# N-gram 모델 종류 
N-gram 모델은 문맥의 크기를 정의하는 N의 값에 따라 분류된다. 일반적인 N-gram 모델은 유니그램(Unigram), 바이그램(Bigram), 트라이그램(Trigram) 등이 있으며, N의 값이 커질수록 더 긴 문맥을 고려하게 된다. 

> N이 더 큰 N-gram 모델(예: 4-gram, 5-gram 등)은 더 긴 문맥을 고려할 수 있지만, 데이터의 희소성(sparsity) 문제가 심해지고, 계산 복잡도가 증가하는 문제가 있다. 따라서 실제 응용에서는 적절한 N의 크기를 선택하는 것이 중요하며, 이는 주로 사용할 데이터의 양과 작업의 특성에 따라 결정된다.
> **문맥이 길다고 큰 N-gram을 쓰는 것 은 아님.** 
더 긴 문맥을 사용하는 모델은 일반적으로 더 정확한 예측을 제공할 수 있지만, 모델의 복잡성, 처리 시간, 필요한 데이터의 양도 함께 증가한다는 점을 고려해야 함.


- **유니그램 (Unigram)**:
유니그램 모델은 N이 1인 경우로, 각 단어의 등장 확률을 독립적으로 계산한다. 즉, 문맥을 전혀 고려하지 않고 각 단어가 개별적으로 나타날 확률만을 사용환다.
유니그램 모델의 단점은 문맥이나 단어의 순서를 전혀 고려하지 않는다는 것.

![image](https://github.com/JinnyKo/NLP-Specialization-Coursera/assets/93627969/a8618beb-8228-4721-a22d-e5992260b7d5)


- **바이그램 (Bigram)**:
바이그램 모델은 N이 2인 경우로, 바로 앞에 오는 하나의 단어를 문맥으로 고려하여 다음 단어의 확률을 계산한다. 바이그램 모델은 인접한 단어 간의 관계를 반영할 수 있다.
바이그램 모델은 유니그램보다는 문맥을 더 고려하지만, 여전히 제한적인 문맥 정보만을 사용.

![image](https://github.com/JinnyKo/NLP-Specialization-Coursera/assets/93627969/eb24a21b-ec42-4973-b0c5-bc5a6e851686)


- **트라이그램 (Trigram)**:
트라이그램 모델은 N이 3인 경우로, 이전의 두 단어를 문맥으로 고려하여 다음 단어의 확률을 계산한다. 트라이그램은 바이그램보다 더 넓은 범위의 문맥을 고려할 수 있다.
더 넓은 문맥을 고려하므로 바이그램보다 일반적으로 더 정확한 예측을 하지만 모델의 복잡도와 요구되는 데이터 양이 증가하는 단점이 있다.

![image](https://github.com/JinnyKo/NLP-Specialization-Coursera/assets/93627969/567bcf5f-39fd-47ea-b0df-1ed5d593aa25)

# Perplexity 
NLP에서 퍼플렉서티(Perplexity)는 언어 모델의 성능을 평가하는 지표 중 하나이다. Perplexity 모델이 얼마나 잘 예측하는지를 수치화한 것으로, 보통 확률적 언어 모델이나 기계 번역 모델의 성능을 평가할 때 사용된다.

직관적으로 Perplexity는 모델이 특정 시퀀스를 볼 때 **평균적으로 고려하는 가능한 선택의 수**를 나타낸다. 예를 들어, Perplexity가 100이라는 언어 모델은 평균적으로 다음에 올 단어로 100개의 가능한 단어들 중에서 선택해야 한다는 것을 의미한다.  퍼플렉서티가 낮을수록 모델이 더 확신을 가지고 단어를 예측한다고 할 수 있으며, 이는 일반적으로 더 좋은 언어 모델을 의미한다. 

> 수학적으로 Perplexity는 주어진 테스트 데이터셋의 문장 확률의 역수를 문장의 길이로 거듭제곱한 값의 평균으로 계산한다.

![image](https://github.com/JinnyKo/NLP-Specialization-Coursera/assets/93627969/a93cb2d4-8a48-4a2b-b8b3-6547bd8f88e5)

![image](https://github.com/JinnyKo/NLP-Specialization-Coursera/assets/93627969/427a3847-ea0a-4259-a0a6-09921c4e7d76)

# Coupus에 없는 단어 다루기(Out of vocabulary words (OOV)) => Open Vocabularies 
#### Open Vocabularies: 
means that you may encounter words from outside the vocabulary, like a name of a new city in the training set. Here is one recipe that would allow you to handle unknown words. 
- Create vocabulary V
- Replace any word in corpus and not in V by <UNK> => 모델이 처리해야하는 단어의 수를 줄여줌
- Count the probabilities with <UNK> as with any other word

> 근데 UNK 처리를 하는것이, 0번 나온 단어를 뜻하는게 아니고, **일정 빈도 이하로 나타나는 단어들을 특별한 토큰인 <UNK> (unknown token)으로 대체** 한다는 것을 의미함.
실제 모델 구축에서 <UNK>를 어떻게 처리할지는 모델러의 목적과 코퍼스의 특성, 모델링하고자 하는 언어 현상에 따라 결정된다.
  

# Smoothing
모델이 훈련 데이터에 나타나지 않은 단어 조합(즉, 빈도수가 0인 N-gram)에 대해 확률을 할당할 수 있도록 하기 위함









