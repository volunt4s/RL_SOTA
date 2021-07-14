# Model Free Prediction, Control

주어진 정책에 대한 가치함수 학습 → 예측 (Prediction)

예측을 토대로 정책 발전 및 최적 정책 학습 → 제어 (Control)

예측

- 몬테 카를로 (Monte Carlo, MC) 예측
- 시간차 예측 (Temporal difference)

제어

- Sarsa
- Sarsa 한계 극복, Off-Policy Q-learning

---

## Monte Carlo Prediction

DP → 환경에 대한 정확한 지식

⇒ 모든 상태에 대해 동시에 계산 진행하므로 계산 복잡도 기하급수적 증가 → 모든 경우의 수를 계산

Retrun to dynamic programming ...

Bellman expectation eqn : $v_{\pi} = E_{\pi}[R_{t+1} + \gamma v_{\pi}(S_{t+1})|S_t=s]$

⇒ 가능한 모든 상황을 고려해 기댓값을 계산

But 적당한 추론을 통해 학습을 해나가면 어떨까?

Policy Iteration → 예측, Policy Improvement → 제어

Monte Carlo → "무작위로 무엇인가를 해본다"

Ex) 종이 위의 원

CODE 참고 → 많은 샘플링 → 오차 적어짐

⇒ Why do this? 기존 방정식 몰라도 해결 가능 ⇒ 어떤 도형이든 넓이 구할 수 있음

Sampling : 원의 넓이를 추정하는 하나의 점을 찍는 것

DP → 기대 방정식을 활용한 "계산" But we need "예측" HOW??

Process

- Terminal state이 존재하는 에피소드를 끝까지 진행 → 그 에피소드에 대한 반환값 계산 → 반환값의 평균 ⇒ 가치함수로 추정
- 상태 변환 확률, 보상 함수 → 몰라도 추정할 수 있음 ⇒ 그래서 "예측"

$$v_{\pi} \risingdotseq \frac{1}{N(s)} \sum_{i=1}^{N(s)}G_i(s)$$

많은 에피소드 진행 → 정확한 가치함수 값 얻을 수 있다고 알려져 있다

몬테 카를로 사용한 가치함수 업데이트 일반적인 식 ⇒

$$V(S) \leftarrow V(S) + \alpha (G(S) - V(S))$$

---

### Temporal-Difference

몬테카를로 → 실시간이 아님 (not Online)

⇒ 에피소드가 끝이 안나거나 끝이 없는 에피소드에서 활용 불가

$$G_t \rightarrow R_{t+1} + \gamma v_{\pi}(S_{t+1})$$

현재 상태에서 정책에 따라 행동을 하나 선택 → 보상을 받고 다음 상태 알게 됨

$$V(S_t)\leftarrow V(S_t) + \alpha(R_{t+1} + \gamma V(S_{t+1}) - V(S_t))$$

이 때, Temporal Difference Error

다른 상태의 가치함수 예측값을 통해 지금 상태의 가치함수를 예측 → Bootstraping

---

### SARSA

GPI (Generalized Policy Iteration) : 정책 평가와 정책 발전을 수행하는 과정

어떤 상태 S(t)에서 탐욕 정책에 따라 A(t) 선택 → (상태 이동), 보상 R(t+1) 받음 → S(t+1), A(t+1) 받음

⇒ S A R S A

현재 가지고 있는 큐함수를 토대로 샘플을 탐욕 정책으로 모으고 그 샘플로 방문한 큐함수를 업데이트

초기 에이전트가 다양한 환경 탐험하는 것이 중요 → $\epsilon$-Greedy 정책

$\epsilon$-Greedy 정책 : 아주 높은 확률로 탐욕 정책, 아주 작은 확률로 탐욕 정책이 아닌 다른 정책

전체적인 Flow

- $\epsilon$-Greedy 정책을 통해 Sample 추출 → Q 함수 업데이트 → 회귀

SARSA의 한계

- 입실론이 상수이기 때문에 갇혀버리는 현상 나타날 수 있음 → 탐욕 정책을 따르지 않았을 때 음의 보상을 받게 되면, 최적 정책임에도 불구하고 잘못된 정책으로 평가해버릴 수 있음

    ⇒ 이를 해결하기 위해 Off-policy control(Q-learning) 등장

---

### Q-learning

Off-policy → 현재 행동하는 정책과 독립적으로 학습

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha(R_{t+1} + \gamma max_{a}Q(S_{t+1}, a') - Q(S_t, A_t))$$

벨만 최적 방정식과 상당히 유사