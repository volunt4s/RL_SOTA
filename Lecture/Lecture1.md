# Intro

### Characteristics of RL

RL이 다른 Learning process와 다른 점

- Supervisor가 없다 → Corrct or not을 모르고 오직 **보상신호(Reward signal)**만 존재
- Feedback이 즉각적이지 않고 Delay
- Agent의 Action → 이어지는 State에 영향 : Markov Property

---

### Rewards

- 보상(Reward) : 스칼라 피드백 신호
- Agent가 해당 step에서 얼마나 **잘 수행하는 지**에 대한 척도
- Def) Reward Hypothesis (보상 가설)
    - RL에서 목표(Goal)는 축적되는 기대 보상의 최댓값으로 표현된다.

---

### Sequential Decision Making

- 목표 : 미래의 총합 보상을 최대화 하는 방향 → 현재의 보상을 미래의 더 큰 보상을 위해 포기할 수 있음.

---

### Agent and Environment

- Agent : 관찰, 행동하는 주체, 행동으로 인한 Environment에게 보상을 부여받음
- Environment : Agent를 감싸고 있는 환경 → Agent에게 보상 부여

---

### History and State

- History : Agent가 관찰, 행동, 보상을 받는 일련의 과정
- History에 따라 다음 행동이 결정됨
- State : 다음으로 어떤 행동을 할지 결정하는 정보
    - State는 History의 함수
    - Environment State
        - Agent에게 observation, reward 부여 → Agent에게 보여지지 않음
        - 마치 아이(Agent)에게 숙제 정답지를 보여주지 않는 것과 같다.
    - Agent State
        - Agent 내부적으로 다음 행동을 할 정보를 받는 상태

---

### Markov State

- Def. Markov

> A state $S_{t}$ is Markov $\Leftrightarrow$ $\mathbb{P}[S_{t+1} | S_{t}] = \mathbb{P}[S_{t+1} | S_{1}, ... , S_{t}$

- 미래의 사건은 과거와 독립적이고 오직 현재에 영향을 받는다.

---

### Fully Observable Environment

- Agent가 직접 Environment state를 관찰한다.
- Agent state = Environment state = Information State
- 이를 Markov decision process라 부름.

---

### Partially Observable Environment

- Agent가 직접적으로 Environment를 관찰할 수 없음.
    - Ex) 로봇의 카메라 센서는 자기 자신의 절대 위치를 알려주지 않음.

---

### Policy, Value function, Model

- Policy : Agent의 행동 양식
- Value function : 보상에 대한 예측
    - State의 좋고 나쁨에 대한 척도
- Model : Environment의 행동 추정
    - 실제 경험전의 Model에서 Planning하여 다음 Action을 결정 → Model-based
    - Model 없이 수많은 Trial & Error로 결정 → Model-free

---

### Conclusion

- 강화학습은 Unsupervised된 환경에서 다양한 Action 시행착오를 통해 Reward를 얻으며 가장 높은 Reward를 받을 수 있는 Action을 학습하는 것을 의미.
- 강화학습에서 학습을 수행하는 주체인 Agent, Agent와 상호작용하며 Reward를 부여하는 Environment가 있다.
- Agent가 놓여진 학습 환경에서 Agent는 주어진 Policy에 따라 최선의 Action을 결정. 이 때 최적의 Policy를 구하는 것이 강화학습의 핵심.
