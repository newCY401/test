# 📚 8차 Quiz (21.08.06)

### 1. 합성곱 연산을 할 때 패딩을 하는 이유는 무엇일까요?

### 2. I<sub>h</sub>, I<sub>w</sub>, K<sub>h</sub>, K<sub>w</sub>, S, P를 이용하여 O<sub>h</sub> 와 O<sub>w</sub>에 대한 식을 세워보세요.

### 3. 풀링 연산과 합성곱 연산의 공통점과 차이점을 설명해보세요.

### 4. 1D 합성곱에서 커널의 사이즈가 달라지는 것이 무슨 의미인지 설명해보세요.

### 5. 1D CNN, BiLSTM 각각을 이용하여 글자 임베딩하는 과정을 간략하게 설명해보세요.

### 6. 다음 코드의 결과를 예측해보세요.

```python
import torch
import torch.nn as nn

inputs = torch.Tensor(1, 1, 28, 28)
conv1 = nn.Conv2d(1, 32, 3, padding=1)
conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
pool = nn.MaxPool2d(2)

out = conv1(inputs)
print(out.shape) #1

out = pool(out)
print(out.shape) #2

out = conv2(out)
print(out.shape) #3

out = pool(out)
print(out.shape) #4

out = out.view(out.size(0), -1) 
print(out.shape) #5

fc = nn.Linear(3136, 10)
out = fc(out)
print(out.shape) #6
```
