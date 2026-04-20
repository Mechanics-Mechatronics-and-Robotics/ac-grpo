# AC-GRPO Methodology (Sparse Reward)

This repository implements a `LunarLander-v2` diagnostic for **Alignment-Certainty Group Relative Policy Optimization (AC-GRPO)** under a **sparse terminal reward** regime.

The goal is to evaluate whether a learned per-timestep certainty signal can stabilize policy optimization and improve robustness when supervision is sparse, delayed, or unreliable.

This diagnostic is designed as a controlled precursor to applying AC-style optimization to large-scale reinforcement learning systems such as **vision-language-action (VLA)** models.

At a high level, the methodology follows the same intuition as **SimpleVLA-RL**:

1. Start from a competent pretrained policy.
2. Perform RL adaptation with a simple binary outcome signal.
3. Evaluate checkpoints explicitly instead of assuming the final checkpoint is best.

> **Note:** This document defines the training protocol and mathematical formulation only. Experiment schedules, run commands, and empirical results must be documented separately.

## Task Setting

The diagnostic environment is `LunarLander-v2` with a discrete action space.

| Quantity              | Value        |
|-----------------------|--------------|
| Observation dimension | 8            |
| Number of actions     | 4            |
| Max episode length    | 1000         |
| Success target        | Safe landing |

Each episode produces a binary outcome:

$$
R_i \in \{0, 1\}
$$

where:
- `1` denotes successful landing
- `0` denotes failure

This binary outcome is the only learning signal used for policy optimization.

## Training Regime and Common Anchor

The sparse-reward experiments are treated as **adaptation from a competent anchor policy**, not sparse-reward learning from scratch.

In the current methodology, all compared methods may start from the same pretrained policy checkpoint. This pretrained checkpoint may itself have been obtained under the original dense environment reward. That is acceptable because:
- The pretrained checkpoint serves only as a **common initialization**.
- All compared methods start from the same anchor.
- The sparse-reward phase tests **adaptation under binary outcome supervision** rather than scratch learning.

This mirrors the intended VLA-style workflow:
`pretraining / supervised initialization → RL adaptation with simple task completion signal`

## Sparse Reward Definition

The training signal is a **terminal-only binary reward**.

At each timestep:

$$
r_t^{\text{train}} = 0 \quad \text{for } t < T
$$
$$
r_T^{\text{train}} = 
\begin{cases}
1 & \text{if success} \\
0 & \text{if failure}
\end{cases}
$$

Therefore, the episode return used by the optimizer is:

$$
G_i^{\text{train}} = R_i
$$

This definition is used identically for `BASELINE`, `AC_LITE`, and `AC_FULL`. No reward shaping is applied. No dense reward is used for learning. This ensures strict comparability between methods and aligns the toy setup with outcome-level RL intuition.

## Environment Reward (Reporting Only)

`LunarLander-v2` internally produces dense environment rewards:

$$
r_t^{\text{env}}
$$

These values are **not used for policy optimization**. They are retained only for reporting and diagnostics:

$$
G_i^{\text{env}} = \sum_{t} r_t^{\text{env}}
$$

This separation enables:
- Consistent sparse learning
- Interpretable performance metrics
- Stable comparison across methods
- Compatibility with binary-outcome RL pipelines

In particular:
- $G_i^{\text{train}}$ is the reward used by PPO/GAE.
- $G_i^{\text{env}}$ is a raw environment performance statistic for analysis only.

## Noise Modes

The pipeline supports three supervision conditions.

| Mode         | Description                                  |
|--------------|----------------------------------------------|
| `CLEAN`      | No corruption                                |
| `REWARD_NOISE` | Terminal success labels are randomly flipped |
| `OBS_NOISE`  | Gaussian noise is added to observations      |

These modes simulate realistic deployment failures in robotics and VLA systems.

## Reward Noise (Sparse Regime)

Reward corruption applies only to terminal outcomes.

With probability:

$$
p = 0.2
$$

a successful episode is converted into a failure label:

$$
R_i^{\text{policy}} = 0 \quad \text{even though} \quad R_i^{\text{raw}} = 1
$$

Implementation logic:
```python
if success and random() < p:
    r_T_train = 0
```

Earlier rewards remain unchanged ($r_t^{\text{train}} = 0$ for $t < T$). Thus, reward noise directly affects the policy gradient in the sparse-reward regime.

Dynamic sampling and AC outcome supervision use the policy outcome label $R_i^{\text{policy}}$, while the raw environment return remains available for diagnostics.

## Observation Noise

Observation corruption is applied at every timestep:

$$
\tilde{s}_t = s_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2 I)
$$

Default: $\sigma = 0.1$. Both the policy and certainty networks receive corrupted observations.

## Methods

The repository compares three training pipelines.

| Method    | Description                                                                     |
|-----------|---------------------------------------------------------------------------------|
| `BASELINE`| PPO-style policy optimization under sparse binary reward                        |
| `AC_LITE` | Baseline policy optimization plus certainty-gated advantages and alignment loss |
| `AC_FULL` | `AC_LITE` plus outcome loss and a discrete-action dispersion proxy              |

The AC extensions are intentionally small. The policy architecture remains unchanged.

## Network Architecture

Policy and certainty networks are fully independent.

**Policy network:**
```
8 → 128 → 128 → 4
```

**Certainty network:**
```
8 → 128 → 128 → 1
```

- **Activation:** ReLU
- **Policy output:** $\pi_\theta(a \mid s)$
- **Certainty output:** $u_t = f_\psi(s_t)$
- **Certainty value:** $c_t = \sigma(u_t)$
- **Clamping:** $c_t \in [10^{-6}, 1 - 10^{-6}]$

No parameters are shared between policy and certainty networks. This isolation prevents degenerate feedback loops where the policy suppresses its own learning signal by manipulating certainty.

## Pretrained Policy Backbone

Training may start from a pretrained policy checkpoint.

Default path:
```text
pretrained_models/lunarlander_baseline_clean_seed42.pt
```

Two operating modes are supported:
- **Fine-tuned policy:** Load pretrained weights and continue updating parameters.
- **Frozen policy:** Load pretrained weights and keep policy fixed while training certainty.

Configuration:
```python
freeze_pretrained_policy: bool
```

This separation allows controlled investigation of two research questions:
1. Can certainty learn meaningful reliability signals on top of a fixed policy?
2. Can certainty-gated optimization improve the policy itself?

The pretrained checkpoint is always considered part of the evaluation trajectory. It should be treated as **checkpoint 0**, and it is allowed to be the best model.

## Baseline Policy Optimization

The baseline uses PPO with Generalized Advantage Estimation.

**Default trainer configuration:**
- Steps per update: 2048
- Batch size: 64
- Optimizer: Adam
- Learning rate: $3 \times 10^{-4}$
- Discount $\gamma$: 0.99
- GAE $\lambda$: 0.95
- Entropy coefficient: 0.01
- Value coefficient: 0.5
- Max gradient norm: 0.5

For sparse-reward adaptation from a pretrained checkpoint, a smaller learning rate may be used in practice. Such changes belong to the experiment configuration, not to the method definition.

GAE:
$$
\hat{A}_t = \text{GAE}(r_t^{\text{train}}, V(s_t), \gamma, \lambda)
$$

PPO objective:
$$
L_{\text{PPO}} = -\min\left(
    r_t(\theta)\hat{A}_t,
    \text{clip}\big(r_t(\theta), 1 - \epsilon_{\text{low}}, 1 + \epsilon_{\text{high}}\big)\hat{A}_t
\right)
$$

Probability ratio:
$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}
$$

Clipping defaults:
```python
epsilon_low  = 0.2
epsilon_high = 0.2
```

The code path also supports asymmetric clipping variants, but symmetric clipping is the default methodology unless explicitly overridden.

## Grouped Rollouts and Dynamic Sampling

Episodes are collected in groups.

Default: `group_size = 4`

Dynamic sampling uses binary outcomes within each group.

Group filtering rule:
```python
keep_group = 0 < successes_in_group < group_size
```

So:
- All-failure groups are uninformative for group-relative updates.
- All-success groups are uninformative for group-relative updates.
- Mixed-outcome groups are preferred.

However, especially in a toy task, mixed groups may be temporarily rare. To prevent total training starvation, the implementation supports:
```python
dynamic_sampling_fallback_on_empty = True
```

If no mixed groups are found in an update window, the sampled groups are used for that update and the fallback event is logged explicitly.

Logged statistics include:
- Discarded group fraction
- Mixed group fraction
- Mean successes per group
- Fallback usage
- Policy entropy
- Gradient norm

This behavior is intentionally close to outcome-driven RL practice while remaining robust enough for a toy diagnostic.

## Rollout Temperature

Sampling temperature modifies exploration:

$$
\pi_T(a \mid s) = \text{softmax}\left(\frac{z_\theta(s)}{T}\right)
$$

Default: `rollout_temperature = 1.0`

Higher temperatures increase exploration by flattening the rollout policy.

## AC-GRPO Certainty Variables

Executed-action probability:
$$
\delta_t = \pi_\theta(a_t \mid s_t)
$$

Certainty:
$$
c_t = \sigma(u_t)
$$

Effective certainty gate:
$$
c_t^{\text{eff}} = c_t(1 - c_{\min}) + c_{\min}
$$

Default: $c_{\min} = 0.3$

Gated advantage:
$$
\hat{A}_t^{\text{AC}} = \text{stopgrad}(c_t^{\text{eff}}) \hat{A}_t
$$

The stop-gradient prevents the policy loss from directly updating the certainty network through the gate. This keeps the certainty signal auxiliary and avoids degenerate coupling.

## Alignment Loss

The alignment loss trains certainty to match the policy's own commitment to the executed action:

$$
L_{\text{align}} = -\delta_t \log c_t - (1 - \delta_t) \log \frac{1 - c_t}{K}
$$

For `LunarLander`: $K = 4$.

The probability $\delta_t$ is detached so this loss trains the certainty network only. This makes alignment a self-consistency objective rather than an additional policy objective.

## Outcome Loss (AC_FULL)

`AC_FULL` adds a trajectory-level Bernoulli likelihood:

$$
L_{\text{outcome}} = -\alpha \left[ R_i \log c_t + (1 - R_i) \log(1 - c_t) \right]
$$

Default: $\alpha = 1.0$

The outcome loss is applied only after episode completion, when the trajectory outcome is known.

## Dispersion Proxy (Discrete Actions)

The original AC-GRPO orbit term is motivated by continuous action geometry. `LunarLander-v2` has discrete actions, so `AC_FULL` uses policy entropy as a discrete dispersion surrogate.

Policy entropy:
$$
H_t = \mathcal{H}\big(\pi_\theta(\cdot \mid s_t)\big)
$$

Dispersion loss:
$$
L_{\text{dispersion}} = \frac{1}{2}\exp(u_t)H_t - \frac{1}{2}\beta u_t
$$

Default: $\beta = 1.0$

Policy entropy is detached so the loss trains the certainty network only. This is not the exact continuous Gaussian orbit likelihood; it is a discrete-action surrogate suitable for the toy setting.

## AC Model Variants

### AC_LITE

`AC_LITE` uses:
1. PPO policy loss with certainty-gated advantages
2. Alignment loss for the certainty network

Certainty objective:
$$
L_{\text{cert}}^{\text{AC-LITE}} = L_{\text{align}}
$$

### AC_FULL

`AC_FULL` uses:
1. PPO policy loss with certainty-gated advantages
2. Alignment loss
3. Outcome loss
4. Entropy-based dispersion proxy

Certainty objective:
$$
L_{\text{cert}}^{\text{AC-FULL}} = L_{\text{align}} + L_{\text{outcome}} + L_{\text{dispersion}}
$$

The policy and certainty optimizers are separate. Certainty-derived gates are detached in the policy loss. Policy-derived quantities are detached in certainty losses where appropriate.

## Logged Quantities

Each run records per-episode and per-step logs.

**Episode-level logs include:**
- Global step
- Episode ID
- Raw return
- Policy outcome label
- Raw success
- Episode length

**Step-level logs include:**
- Timestep
- Policy entropy
- Executed-action probability $\delta_t$
- Certainty $c_t$

**Update-level logs include:**
- Loss
- Gradient norm
- Mean policy entropy
- Kept steps
- Fallback flag

**Per-seed summaries include:**
- Method
- Mode
- Seed
- Total steps
- Full configuration
- Pretrained policy path
- Frozen policy flag
- Reward-noise semantics
- Certainty-gate parameters

## Evaluation Protocol

Evaluation always uses a greedy policy.

Checkpoint selection is performed by evaluating **all saved checkpoints**, including the pretrained anchor checkpoint, on fixed held-out episodes. The final checkpoint is not assumed to be optimal.

Held-out evaluation is mode-matched:
- `CLEAN` checkpoints are evaluated on `CLEAN` held-out episodes.
- `OBS_NOISE` checkpoints are evaluated on `OBS_NOISE` held-out episodes.
- `REWARD_NOISE` checkpoints are evaluated on `REWARD_NOISE` held-out episodes.

**Primary metrics:**
- Success rate
- Raw environment return
- Best checkpoint performance
- Seed mean and standard deviation

This protocol is intentionally checkpoint-centric, since outcome-driven RL can degrade after early gains.

## Design Principle

The experiment isolates a single causal variable: **certainty-gated policy optimization**.

All methods share:
- Identical sparse binary reward
- Identical optimizer family
- Identical architecture
- Identical rollout pipeline
- Identical pretrained anchor when used

Only the certainty mechanism differs. This ensures that any observed performance difference can be attributed to certainty-aware optimization rather than reward engineering.