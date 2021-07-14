# CHAP2. Markov Desicion Process

### Markov Process

- Markov Property → 미래의 사건은 오직 현재의 영향을 받는다.
- Markov state $S_{t}, S_{t+1}$사이에 State transition probability(상태 전이 확률) 존재.
    - t에서 t+1로 일어날 확률?
- 이러한 Markov State들의 연속을 **Markov Process(Markov Chain)**라고 함.

우리 일상의 모든 행동들은 Markov chain으로 나타낼 수 있음. 예를 들어 아침에 일어나서(State 1) 물을 마실(State 2) 확률이 0.5, 화장실에 갈(State 3) 확률 0.5 등으로 나타냄.
이 때 한 State에서 다른 State로 넘어가는 모든 확률의 합은 1이어야 함.

- Markov chain에서 각각 State의 연결을 **Episode**라고 함.
- Markov chain에 해당하는 Transition Probability를 Matrix 형태로도 표현 가능

---

### Markov Reward Process

- 일련의 Markov chain에 Reward 개념을 추가한 것 → MRP
    - 각 State마다 부여받는 Reward 값이 있음.
    - 이 때 주어진 t에서 t+1의 Reward를 받는 것을 **Immediate Reward**라고 함.
- Return
    - 주어진 시간 동안의 Discounted Reward의 총합
    - Discount
        - Immediate Reward를 제외한 나머지 Futre Reward는 Discount 함.
        - Why?
            - 현재 가치와 미래 가치의 가중을 어느정도 두느냐에 따라 궁극적 가치가 달라짐.

            은행 이자처럼 현재보다 미래의 가치를 더 두는 상황이 좋을 수도 있음. ⇒ Discount Factor를 0~1 사이에서 조정

- Value Function
    - MRP에서의 Value Function은 해당 State에서 Return의 기댓값
    - 각 Episode마다 값이 달라질 수 있음. → Episode Sequence의 각 State에 해당하는 Reward 값이 다르기 때문

---

### Bellman Equation for MRP

- MRP에서의 Value Function은 2가지로 분리할 수 있음
    1. Immediate Reward
    2. 1. 이후의 일련의 State의 Discounted Value
- 이를 정리한 것이 MRP에서의 Bellman Equation.
- Bellman Equation은 Linear equation이므로 해석적 풀이가 가능하지만 오직 작은 size에서 가능
    - 이를 Iterative 하게 풀어내는 것이 Dynamic Programming, Monte-Carlo Evaluation 등이 있음

---

### Markov Decision Process

- MRP에서 Action(Decision)을 추가한 것이 MDP
- MRP의 정의(State transition probability matrix, Reward function)에 Action 추가
- State가 Action을 함에 따라 Transition하게 됨.
- Policy
    - 주어진 State에서 어떤 Action을 하게 될 확률 → Agent의 행동을 결정함.
    - Policy는 오직 현재의 State에 의존함. → Markov Property
- Value Function
    - State-value function
        - 주어진 Policy를 바탕으로, 어떤 State에서의 Return의 기댓값
    - Action-value function
        - 주어진 Policy를 바탕으로, 어떤 State에서 Action을 했을 때 Return의 기댓값

---

### Bellman Expectation Equation

- Bellman equation과 마찬가지로, MDP에서 State value function, Action value function은 현재의 즉각 보상, 이후 일련의 Discounted value로 나눌 수 있음.
- Bellman Expectation Eqn. for State value function
    - 현재 State에서 각 Action을 할 확률과 그 Action으로 받는 Return의 곱
- Bellman Expectation Eqn. for Action value function
    - Action으로 인한 즉각 보상 + 다음 State에서의 Discounted Value function

---

### Optimal Value function

- 모든 Policy에 해당하는 Value function의 최댓값 → Optimal Value function
- 마찬가지로 Action Value function의 최댓값 → Optimal Action value function
- 결국 MDP에선 **Optimal Value function을 찾는 것이 핵심**
- Optimal policy
    - 모든 MDP에선 Optimal Value function을 만족하는 Optimal policy가 존재한다.
    - 어떤 Action이 최대의 Action Value function을 가질 때 → Optimal policy
        - 이를 시행착오하며 찾아내는 것이 RL
