# In this Week
![image](https://github.com/JinnyKo/NLP-Specialization-Coursera/assets/93627969/90c40b4a-01ea-41f2-8395-500682618715)

# Basic Word Representations (단어 표현 방법) 
- **Integers**
 각 단어에 고유한 정수를 할당하는 방법.예를 들어 "happy"라는 단어에 621이라는 숫자를 할당한다. 하지만 단어 사이에 수학적 관계가 없음에도 불구하고 숫자 크기에 의한 임의의 순서나 관계가 생긴다.

- **One-hot vectors**
 단어를 벡터로 표현하는데, 벡터의 크기는 어휘(Vocabulary)의 크기와 같고, 해당 단어의 인덱스 위치에만 1을 두고 나머지는 모두 0으로 채우는 방법이다. 예를들어 "happy"라는 단어에 대한 원-핫 벡터는 "happy"의 인덱스에 해당하는 위치에만 1을 두고, 나머지는 0으로 이루어져 있다.
=> **고차원의 희소한 벡터**
ex) 예를 들어, 단어 "사과"를 표현할 때, 단어 집합의 크기가 10,000이라면 "사과"에 해당하는 인덱스 위치만 1이고 나머지는 0인 10,000차원의 벡터가 만들어진다.

- **Word embedding**
각 단어를 저차원의 연속적인 값으로 이루어진 벡터로 표현하는 방법. 단어 임베딩은 단어 간의 의미적 관계를 포착하고, 단어 사이의 유사성을 수치적으로 표현할 수 있다.
=> **저차원의 밀집된 벡터**
ex) 예를 들어, "사과"라는 단어를 100차원의 벡터로 표현할 수 있다. 이 벡터는 각 차원이 해당 단어의 특정 의미적 특징을 나타낸다. [0.2,0.5,−0.1,...,0.3(n=100)]

![image](https://github.com/JinnyKo/NLP-Specialization-Coursera/assets/93627969/24086256-d1fe-40ee-bcc0-484f3f296f26)

# Embeddings (How to create word Embeddings) 
**Self-supervised learning** 
기계 학습에서 사용되는 한 종류의 학습 방법이다. 이 방법은 레이블이 없는 데이터로부터 특징을 학습하는 방식으로, 데이터의 내재된 구조를 이용하여 학습하는 것을 의미한다. 
### Embedding 생성하는 모델

> #### Word2ve
- CBOW: 주변 단어들을 사용하여 중심 단어를 예측.
- Skip-gram: 중심 단어를 사용하여 주변 단어를 예측.

>#### BERT, Bidirectional Encoder Representations from Transformers:
- 양방향 Transformer 인코더를 사용하여 단어나 문장을 임베딩하는 데 사용된다.
- 사전 학습된 언어 모델로, 대규모 텍스트 데이터를 사용하여 사전 학습된다.
-  단어 임베딩을 생성할 뿐만 아니라, 문맥을 고려한 문장 임베딩을 생성할 수 있다.
  
> #### ElMo (Embeddings from Language Models):
- ElMo는 양방향 LSTM (Long Short-Term Memory) 기반의 언어 모델을 사용하여 단어의 표현을 학습한다.
- ElMo는 주어진 문맥에서 단어의 의미를 파악하기 위해 단어의 다양한 의미를 포착하는 데 중점을 둔다.
- ElMo는 다층의 양방향 LSTM을 사용하여 단어의 다양한 의미적 표현을 학습하며, 이를 통해 단어 임베딩을 생성한다.







