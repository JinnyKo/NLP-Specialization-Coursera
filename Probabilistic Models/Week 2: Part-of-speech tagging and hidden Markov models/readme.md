## 마르코프 체인 (Markov Chains) 
> 미래의 상태가 오직 현재 상태에만 의존하고 과거 상태와는 무관한 확률 과정을 설명하는 수학적 모델..
> 간단히 말해서 **마르코프 체인에서는 다음 상태로의 전이 확률이 현재 상태에만 의존하며, 이전의 경로나 상태에는 영향을 받지 않는다**고 할 수있다.
> 이 개념은 자연어 처리에서 뿐만 아니라 경제학, 게임이론 등에서도 자주 쓰인다.
> 자연어 처리에서는 단어 또는 문자의 시퀀스를 생성할 때 마르코프 체인을 사용할 수 있는데, 단어나 문자의 연쇄가 마르코프 프로세스를 따른다. 

>거대모델을 학습시키는걸 기반으로 생각하니 계속 의문점이 들었다. GPT와 같은 거대모델을 학습시킬 때는 이 마르코프 체인을 따르지 않는다는 점.

- **마르코프 체인** : 간단한 예측, 텍스트 생성, 날씨 변화 모델링 등 제한된 정보를 바탕으로 하는 예측에 주로 사용됨.
- **트랜스포머** : 고급 자연어 처리 작업, 기계 번역, 텍스트 요약, 질문 응답 시스템 등 복잡한 언어 모델링과 관련된 작업에 사용됨.
  
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

