## 마르코브체인 (Markov Chains) 
> 미래의 상태가 오직 현재 상태에만 의존하고 과거 상태와는 무관한 확률 과정을 설명하는 수학적 모델..
> 간단히 말해서 마르코프 체인에서는 다음 상태로의 전이 확률이 현재 상태에만 의존하며, 이전의 경로나 상태에는 영향을 받지 않는다고 할 수있다.
> 이 개념은 자연어 처리에서 뿐만 아니라 경제학, 게임이론 등에서도 자주 쓰인다.
> 자연어 처리에서는 단어 또는 문자의 시퀀스를 생성할 때 마르코프 체인을 사용할 수 있는데, 단어나 문자의 연쇄가 마르코프 프로세스를 따른다. 

```
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# 필요한 nltk 리소스를 다운로드
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# 샘플 문장
sentence = "Markov chains can be used for various applications in natural language processing."

# 문장을 단어로 토큰화하고 각 단어에 대해 품사를 태깅
tokens = word_tokenize(sentence)
tagged_tokens = pos_tag(tokens)

print(tagged_tokens)

```

