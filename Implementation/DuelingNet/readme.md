# Dueling Network Architectures for Deep RL

Dueling Network Architectures Algorithm implementation with OpenAI GYM LunarLander environment

### TODO

- [x]  Paper review
- [x]  Code implementation
- [x]  Optimizing Code

---

## Paper review

### Abstract

이전까지 Deep Neural net과 RL을 결합하려는 많은 시도가 있었음. (CNN, LSTMs, Auto-encoder) → Conventional한 방법이라고 표현

새로운 Neural net Architecture 고안 → 2개의 분리된 Estimator

- estimate state value function
- estimate action advantage function

이러한 방법은 RL 알고리즘을 변경하지 않고서 일반적으로 학습할 수 있음 → 더 나은 결과도 보여줌

### Introduction

기존 DQN Neural net과 달리 state value랑 action advantage로 분리되는 구조 설계

이 두 흐름이 나중에 합쳐서 state action value Q를 내보낸다. → Single Q learning으로 이해하면 됨

직관적으로 이러한 구조는 어떤 state가 값진 것인지 학습할 수 있게 됨

- Atari Enduro game에서, value function은 길의 끝과 점수판에 대해 집중적으로 활성화
- advantage는 당장 눈 앞의 장애물에 대해 집중적으로 활성화

결과적으로 이러한 구조가 더 빨리 정확한 action으로 수정할 수 있게 하는 장점이 있음

### Background

DDQN + Priortized Experience Reaplay

### The Dueling Network Architecture

많은 state에서, action decision estimation이 크게 필요하지 않다

- 예를 들어, 당장 눈 앞의 장애물을 피하기 위해서 왼쪽, 오른쪽 action을 선택해야 하지만 그냥 있는 상황에서는 action 선택이 큰 영향을 미치지 않음

기존 DQN 구조에서, CNN + Fully connected 다음에 2개의 stream으로 나뉘어짐. → value, advantage

- 마지막 output은 action value Q이기 때문에, 동일한 Model free RL 적용 가능
- 알고리즘은 동일하게 가져가면서 성능 향상을 기대할 수 있음

기본 알고리즘 모티베이션

- Q = V + A
- A의 기댓값이 0 이거나, A가 최적 policy를 따를 때 A = 0 임을 이용 ⇒ 2가지 방법이 있음
- 기댓값 버전 : A - 평균 A
- 최적 버전 : A - max A

DDQN → 최적 action과 그 외 action과의 Q 차이가 0.04정도로 미미했다

- 약간의 노이즈에도 쉽게 영향 받을 수 있음
- Dueling net은 state과 action을 분리하기 때문에 이와같은 현상을 잘 버틴다.

### Experiment

간단한 환경(Corridor environment)에서 실험

- Dueling net이 전반적으로 더 빠르게 optimal policy에 수렴했고, 특히 action 가짓수가 많아질 수록 눈에 띄는 향상을 보임

### I learned

학습할 때 epsilon 크게 가져가도 괜찮은 듯 → Testing 따로 가져가고 학습할 땐 overfitting 방지

- 5% → 8% 훨씬 안정적인 학습

Regularization 필수 → 안정적

구조 간단하게 바꾸는 것으로도 바꿔도 좋은 성능 향상

Hyperparameter 튜닝이 어렵다 (Engineering에 시간을 많이 쏟아야...)