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

## Classical Methods
> #### Word2ve 
- CBOW: 주변 단어들을 사용하여 중심 단어를 예측.
- Skip-gram: 중심 단어를 사용하여 주변 단어를 예측.
#### Continuous bag-of-words (CBOW):
the model learns to predict the center word given some context words.
#### Continuous skip-gram / Skip-gram with negative sampling (SGNS): 
the model learns to predict the words surrounding a given input word.
#### Global Vectors (GloVe) (Stanford, 2014): 
factorizes the logarithm of the corpus's word co-occurrence matrix,  similar to the count matrix you’ve used before.
#### fastText (Facebook, 2016): 
based on the skip-gram model and takes into account the structure of words by representing words as an n-gram of characters. It supports out-of-vocabulary (OOV) words.

## Deep learning, contextual embeddings
>#### BERT, Bidirectional Encoder Representations from Transformers:
- 양방향 Transformer 인코더를 사용하여 단어나 문장을 임베딩하는 데 사용된다.
- 사전 학습된 언어 모델로, 대규모 텍스트 데이터를 사용하여 사전 학습된다.
-  단어 임베딩을 생성할 뿐만 아니라, 문맥을 고려한 문장 임베딩을 생성할 수 있다.
  
> #### ElMo (Embeddings from Language Models):
- ElMo는 양방향 LSTM (Long Short-Term Memory) 기반의 언어 모델을 사용하여 단어의 표현을 학습한다.
- ElMo는 주어진 문맥에서 단어의 의미를 파악하기 위해 단어의 다양한 의미를 포착하는 데 중점을 둔다.
- ElMo는 다층의 양방향 LSTM을 사용하여 단어의 다양한 의미적 표현을 학습하며, 이를 통해 단어 임베딩을 생성한다.

# Sliding Windows of words
전테 텍스트 데이터에서 문맥(주변 단어, 중심 단어) 을 정의하기위해서 사용함. 
1. 문장이나 문서를 특정 크기의 윈도우로 나눈다.
2. 윈도우를 한 단계씩 이동하면서 중심 단어와 주변 단어를 결정한다.
3. 중심 단어와 주변 단어를 사용하여 모델을 학습하거나 다른 작업을 수행한다.

```
def sliding_window(sentence, window_size=3):
    words = sentence.split()  # 문장을 단어로 분할
    
    contexts = []
    targets = []
    
    for i in range(len(words)):
        center_word = words[i]
        start = max(0, i - window_size // 2)  # 윈도우 시작 인덱스
        end = min(len(words), i + window_size // 2 + 1)  # 윈도우 끝 인덱스
        context = [words[j] for j in range(start, end) if j != i]  # 중심 단어를 제외한 주변 단어들
        
        if len(context) == window_size - 1:  # 윈도우 크기와 일치하는 주변 단어가 있는 경우
            contexts.append(context)
            targets.append(center_word)
    
    return contexts, targets
```

> 다음은 교재의 예시 (yield 사용 하는 것 다시 보려고)  
```
def get_windows(words,C):
    i=C
    While i <len(words) - C:
         center_word = words[i]
         context_words = words[(i-C) :i] + words[(i+1) : (i+C+1)]
         yield context_words, center_word
         i += 1 
```

### 틀렸던 Quiz 
> Q: You are designing a neural network for a CBOW model that will be trained on a corpus with a vocabulary of 8000 words. If you want it to learn 400-dimensional word embedding vectors, what should be the sizes of the input, hidden, and output layers?
> #### The answer is input 8000 hidden 400 output 8000.
> CBOW 모델은 hidden layer 가 없는 것 아닌가....? **그래서 input 8000, output 8000** 이라고 생각함.

### CBOW 모델 
> CBOW 모델은 중심 단어(center word)를 기반으로 주변 단어(context words)를 예측하는 모델. 따라서 입력은 주변 단어들의 원-핫 인코딩 벡터이고, 출력은 중심 단어의 원-핫 인코딩 벡터이다. 모든 CBOW 모델에서는 입력과 출력 간에 은닉층(hidden layer)이 있는 것은 아님.

![image](https://github.com/JinnyKo/NLP-Specialization-Coursera/assets/93627969/733d5bb0-633b-4407-8910-437a159b0900)

1. **입력층 (Input layer):**
   - 입력 벡터 \( x \)는 크기가 \( V \times 1 \)인 벡터이다. 여기서 \( V \)는 어휘 사전의 크기를 의미한다. 이 벡터는 원-핫 인코딩 방식으로 주변 단어들을 나타낸다.

2. **첫 번째 행렬 곱셈 (First matrix multiplication):**
   - \( W_1 \)는 첫 번째 가중치 행렬로, \( N \times V \) 차원을 가진다. 여기서 \( N \)은 은닉층의 크기이다.
   - \( b_1 \)는 첫 번째 편향 벡터로, \( N \times 1 \) 차원을 가진다.
   - \( z_1 = W_1x + b_1 \) 연산을 통해 입력 벡터 \( x \)에 첫 번째 가중치 행렬 \( W_1 \)을 곱하고 편향 \( b_1 \)을 더합니다. 이 결과는 \( N \times 1 \) 차원의 벡터 \( z_1 \)을 얻는다.

3. **활성화 함수 (Activation function):**
   - 활성화 함수로 ReLU(Rectified Linear Unit)가 사용된다. 이는 \( h = ReLU(z_1) \)로 표현되며, \( z_1 \)의 모든 음수 값을 0으로 설정한다. 결과 \( h \)는 여전히 \( N \times 1 \) 차원을 유지한다.

4. **두 번째 행렬 곱셈 (Second matrix multiplication):**
   - \( W_2 \)는 두 번째 가중치 행렬로, \( V \times N \) 차원을 가진다,
   - \( b_2 \)는 두 번째 편향 벡터로, \( V \times 1 \) 차원을 가진다.
   - \( z_2 = W_2h + b_2 \) 연산을 통해 은닉층 벡터 \( h \)에 두 번째 가중치 행렬 \( W_2 \)를 곱하고 편향 \( b_2 \)를 더한다. 이 결과는 \( V \times 1 \) 차원의 벡터 \( z_2 \)를 얻는다.

5. **출력층 (Output layer) 및 소프트맥스 (Softmax):**
   - 마지막으로, \( \hat{y} = \text{softmax}(z_2) \) 연산을 수행. 여기서 소프트맥스 함수는 \( z_2 \) 벡터의 각 요소를 자연상수 \( e \)의 지수로 변환한 후, 모든 요소의 합으로 각 값을 나누어 줍니다. 이는 벡터의 각 요소를 확률로 변환해 주며, 결과 벡터 \( \hat{y} \)는 모델이 예측한 단어의 분포를 나타낸다. 크기는 \( V \times 1 \)이다.


### 예를 들어
"The cat sat on the ____" 라는 문장을 가지고 있고, 빈칸에 들어갈 단어를 예측하고 싶다고 할 때, 어휘 사전에는 'the', 'cat', 'sat', 'on', 'mat', 'dog'이라는 단어들이 있고, 이를 원-핫 인코딩으로 표현한다면 각각의 벡터 V는 6차원이 된다 (사전에 있는 단어의 수가 6개이기 때문).

원-핫 인코딩된 벡터의 예:
- 'the' -> [1, 0, 0, 0, 0, 0]
- 'cat' -> [0, 1, 0, 0, 0, 0]
- ...

1. **입력층**:
   - 입력 \( x \)는 'the', 'cat', 'sat', 'on'에 해당하는 원-핫 벡터의 평균이 된다.
   > 입력 벡터 x (원-핫 인코딩된 'the', 'cat', 'sat', 'on'의 평균):
   > [0.25, 0.25, 0.25, 0.25, 0.0, 0.0]

2. **첫 번째 행렬 곱셈**:
   - 가중치 행렬 \( W_1 \)과 벡터 \( x \)를 곱한다. 예를 들어, \( W_1 \)이 3x6 (은닉층 크기 \( N \)이 3이라고 가정) 행렬이라면, 결과 \( z_1 \)은 3차원 벡터가 된다.
     > 첫 번째 행렬 곱셈 결과 z1
     > [[1.09592173],
        [0.84959098],
        [0.9643557 ]]


3. **ReLU 활성화 함수**:
   - \( z_1 \)에 ReLU를 적용하면 음수 값은 0이 되고, 그 결과 \( h \) 또한 3차원 벡터가 됨.
    > ReLU 활성화 함수를 적용한 후의 은닉층 벡터 h (음수 제거):
    > [[1.09592173],
      [0.84959098],
       [0.9643557 ]]


4. **두 번째 행렬 곱셈**:
   - 이제 \( h \)에 가중치 행렬 \( W_2 \) (6x3 크기)를 곱하고, 편향 \( b_2 \)를 더한다. 결과적으로 \( z_2 \)는 6차원 벡터가 된다.
     > 두 번째 행렬 곱셈 결과 z2
     > [[1.19453344],
       [1.4814902 ],
       [1.6068424 ],
       [0.90781876],
       [3.54919954],
       [1.3354382 ]]
     
5. **Softmax**:
   - 마지막으로, \( z_2 \)에 소프트맥스 함수를 적용하여 각 단어에 대한 확률 분포를 얻는다. 예를 들어, 만약 소프트맥스 함수가 [0.1, 0.1, 0.1, 0.1, 0.5, 0.1]이라는 결과를 출력했다면, 모델은 'mat'이 빈칸에 들어갈 확률이 가장 높다고 예측하는 것이다 ('mat'에 해당하는 인덱스가 5번째이므로).
     > Softmax 함수를 적용하여 얻은 예측 확률 분포 y^
     > [0.06142763, 0.08184412, 0.09277421, 0.04611531, 0.64711617, 0.07072255]

이렇게 CBOW 모델은 주변 단어들을 사용하여 빈칸에 들어갈 단어를 예측하게 된다. 이 과정에서 학습되는 가중치 행렬은 단어의 벡터 표현을 생성하는 데 사용되고, 이 벡터들은 단어의 의미를 수치적으로 나타내는 방법을 학습한다.


# Architecture of the CBOW Model: Activation Functions
![image](https://github.com/JinnyKo/NLP-Specialization-Coursera/assets/93627969/af4ba8f5-18b8-40a2-a64c-84edb3756357)

- ### Relu Functions
  인공 신경망에서 사용되는 활성화 함수 중 하나. ReLU 함수는 입력값이 양수일 경우에는 그대로 출력하고, 음수일 경우에는 0으로 출력한다. 이를 통해 비선형성을 도입하여 신경망이 더 복잡한 패턴을 학습할 수 있도록 한다.

가정:
- CBOW 모델은 주변 단어 2개를 입력으로 받고, 중심 단어를 예측
- 각 단어는 3차원의 임베딩 벡터로 표현
- 각 입력 임베딩 벡터는 가중치와 편향이 적용
- ReLU 함수는 활성화 함수로 사용

입력:
- 주변 단어 2개로 "love"와 "eat"을 예시로
- 각 단어는 다음과 같은 임베딩 벡터로 표현
  - "love": [0.1, 0.2, 0.3]
  - "eat": [-0.2, 0.4, -0.1]

가중치와 편향:
- 가중치는 각 입력 임베딩 벡터에 곱해지고, 편향은 더해짐
- 가중치는 다음과 같이 설정
  - "love"의 가중치: [0.5, -0.3, 0.2]
  - "eat"의 가중치: [0.2, 0.1, -0.4]
- 편향은 모두 0으로 가정

계산:
1. "love"와 "eat"의 임베딩 벡터에 각각 가중치를 곱한 후 더한다.
   - \( (0.1 \times 0.5) + (-0.2 \times 0.2) = 0.05 - 0.04 = 0.01 \)
   - \( (0.2 \times -0.3) + (0.4 \times 0.1) = -0.06 + 0.04 = -0.02 \)
   - 
2. ReLU 함수에 적용
   - 첫 번째 값은 양수이므로 그대로 출력
   - 두 번째 값은 음수이므로 0으로 출력
   
결과:
- 출력값은 [0.01, 0]

![image](https://github.com/JinnyKo/NLP-Specialization-Coursera/assets/93627969/8388f158-ffa1-4196-9cf6-e5461a246e4b)

- ### Softmax Function
  신경망의 출력을 확률 분포로 변환하는 함수. 주로 다중 클래스 분류 문제에서 출력층에서 사용된다. 소프트맥스 함수는 각 출력값을 0과 1 사이의 값으로 변환하고, 모든 출력값의 합이 1이 되도록 정규화한다.
  
가정:
- 신경망은 다중 클래스 분류를 수행
- 출력층에서는 소프트맥스 함수가 사용
- 출력층의 출력값은 [2, 1, 0.5]으로 가정

계산:
1. 소프트맥스 함수에 입력값 
   - \( e^2, e^1, e^{0.5} \)를 각각 계산
   - 이때 \( e \)는 자연상수인 오일러 수
2. 각 값의 지수 함수 값을 모두 더함.
   - \( e^2 + e^1 + e^{0.5} \)를 계산
3. 각 값의 지수 함수 값을 모든 값의 합으로 나누기
   - 각 값의 지수 함수 값 / 모든 값의 합을 계산
   - 이를 통해 각 클래스에 대한 확률을 얻을 수 있다.

결과:
- 소프트맥스 함수를 거친 후의 출력값은 [0.576, 0.285, 0.139]. 이는 각 클래스에 속할 확률이다. 

# Forward Propagation 
![image](https://github.com/JinnyKo/NLP-Specialization-Coursera/assets/93627969/6ca6901d-4515-43ce-bf4c-77af81e7b852)

# Bcakpropagation 
역전파(Backpropagation):
정방향 전파에서는 입력층부터 출력층까지의 예측값을 계산한다.. 이후, 역전파 단계에서 오차를 최소화하기 위해 각 층의 가중치를 업데이트한다. 역전파 알고리즘은 오차를 역방향으로 전파하여 각 층의 가중치에 대한 그래디언트를 계산한다.

오차(손실 함수): \( E = \frac{1}{2}(y_{\text{true}} - y_{\text{pred}})^2 \)
가중치의 업데이트: \( w_{\text{new}} = w_{\text{old}} - \eta \frac{\partial E}{\partial w} \)
여기서 \( \frac{\partial E}{\partial w} \)는 손실 함수에 대한 가중치의 편미분 값.  역전파 알고리즘은 이 값을 계산하여 각 층의 가중치를 업데이트한다.

# Gradient 
그래디언트는 손실 함수의 모든 파라미터에 대한 편미분 값을 모은 벡터. 경사 하강법과 같은 최적화 알고리즘에서는 이 그래디언트를 사용하여 손실 함수를 최소화하는 방향으로 파라미터를 업데이트한다.
그래디언트: \( \nabla E = \left[ \frac{\partial E}{\partial w_1}, \frac{\partial E}{\partial w_2}, \dots, \frac{\partial E}{\partial w_n} \right] \)
여기서 \( \frac{\partial E}{\partial w_i} \)는 손실 함수에 대한 각 파라미터 \( w_i \)의 편미분 값. 경사 하강법은 이 그래디언트를 사용하여 손실 함수를 최소화하는 방향으로 파라미터를 업데이트힌다.

# Extracting Word Embedding Vectors 
#### 중요한건 train이 끝난 후에 결과로써 Word Embedding 값을 가져와야 한다. 근데 이건 학습 과정에서 직접적으로 출력되는 어떤 값이 아니고, 이 프로세스의 부산물 같은 것이다. 
- Word Embedding을 학습하는 과정에서는 각 단어에 대한 임베딩 벡터가 최적화되는 과정에서 업데이트된다.  학습이 끝난 후에는 이러한 학습된 임베딩 벡터를 추출하여 저장할 수 있다. 보통 이러한 임베딩 벡터들은 모델의 일부로 저장되거나 따로 파일로 저장된다.

### 방법 

- 모델의 가중치(weight)를 확인:
Word Embedding은 신경망 모델의 가중치 중에 해당하는 부분이다. 따라서 학습된 모델에서 해당 가중치를 확인하고 추출할 수 있다.
예를 들어, Word2Vec이나 GloVe와 같은 모델의 경우, 학습된 단어 임베딩은 모델의 Embedding 레이어에 해당하는 가중치로 저장된다.

-모델의 API를 사용하여 추출:
몇몇 딥 러닝 프레임워크에서는 학습된 모델에서 Word Embedding을 추출할 수 있는 API를 제공한다. 이를 사용하여 학습된 임베딩 벡터를 추출할 수 있다.
예를 들어, TensorFlow나 PyTorch와 같은 프레임워크에서는 모델의 Embedding 레이어에 직접 접근하여 임베딩 벡터를 추출할 수 있다.

- pretrained된 모델을 사용:
사전에 훈련된 Word Embedding 모델을 사용하는 경우, 해당 모델의 임베딩 벡터를 직접 다운로드하여 사용할 수 있다.
예를 들어, GloVe나 Word2Vec과 같은 사전 훈련된 모델은 미리 학습된 단어 임베딩을 제공하며, 이를 다운로드하여 사용할 수 있다.

#  Intrinsic evaluation 
![image](https://github.com/JinnyKo/NLP-Specialization-Coursera/assets/93627969/8afc7ccb-f19e-4a76-99d2-f826e3f17382)


# Extrinsic Evaluation 
![image](https://github.com/JinnyKo/NLP-Specialization-Coursera/assets/93627969/d20ff2ed-52a2-4fb0-9f9b-0805b18100e6)















