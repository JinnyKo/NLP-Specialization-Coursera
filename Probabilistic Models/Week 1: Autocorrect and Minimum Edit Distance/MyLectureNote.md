### Learn about autocorrect, minimum edit distance, and dynamic programming, then build your own spellchecker to correct misspelled words!

- Word probabilities
- Dynamic programming
- Minimum edit distance
- Autocorrect

## 1. What is autocorrect? 
잘못 입력된 단어나 문장을 자동으로 수정해 주는 기능을 말합니다.
이 기능은 텍스트를 입력할 때 오타나 철자 오류를 바로잡아 줍니다.
예를 들어, 스마트폰이나 컴퓨터에서 메시지를 작성할 때 '안녕하세ㅛ'라고 타이핑하면, '안녕하세요'로 자동 수정하는 것이 autocorrect 기능의 예입니다.

## 2. What is Minimum edit distance algorithm
두 문자열 사이의 '최소 편집 거리'를 계산하는 알고리즘입니다. 최소 편집 거리란 한 문자열을 다른 문자열로 변환하기 위해 필요한 최소한의 삽입, 삭제, 대체 작업의 횟수를 의미합니다. 예를 들어, 'cat'을 'cut'으로 바꾸려면 'a'를 'u'로 대체하는 한 번의 작업이 필요하므로, 이 두 단어의 최소 편집 거리는 1입니다.
이 알고리즘은 다양한 분야에서 활용되며, 특히 자연어 처리에서 오타 교정, 단어 유사도 측정 등에 사용됩니다. Autocorrect 기능에서도 이 알고리즘을 활용하여 사용자가 입력한 단어와 유사한 올바른 단어를 찾아 제안할 수 있습니다.

## Assignment To implement autocorrect in this week's assignment, you have to follow these steps: 

- Identify a misspelled word
- Find strings n edit distance away: (these could be random strings)
- Filter candidates: (keep only the real words from the previous steps)
- Calculate word probabilities: (choose the word that is most likely to occur in that context)

## Dynamic Programming 

