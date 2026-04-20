# AC-GRPO Methodology (Sparse Reward)

This repository implements a `LunarLander-v2` diagnostic for **Alignment-Certainty Group Relative Policy Optimization (AC-GRPO)** under a **sparse terminal reward** regime.

The goal is to evaluate whether a learned per-timestep certainty signal can stabilize policy optimization and improve robustness when supervision is sparse, delayed, or unreliable.

This diagnostic is designed as a controlled precursor to applying AC-style optimization to large-scale reinforcement learning systems such as **vision-language-action (VLA)** models.

At a high level, the methodology follows the same intuition as **SimpleVLA-RL**:

1. Start from a competent pretrained policy.
2. Perform RL adaptation with a simple binary outcome signal.
3. Evaluate checkpoints explicitly instead of assuming the final checkpoint is best.

> **Note:** This document defines the training protocol and mathematical formulation only. Experiment schedules, run commands, and empirical results must be documented separately.

---

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

where `1` denotes successful landing and `0` denotes failure. This binary outcome is the only learning signal used for policy optimization.

---

## Training Regime and Common Anchor

The sparse-reward experiments are treated as **adaptation from a competent anchor policy**, not sparse-reward learning from scratch.

All compared methods start from the same pretrained policy checkpoint. This pretrained checkpoint may itself have been obtained under the original dense environment reward. That is acceptable because:
- The pretrained checkpoint serves only as a **common initialization**.
- All compared methods start from the same anchor.
- The sparse-reward phase tests **adaptation under binary outcome supervision** rather than scratch learning.

This mirrors the intended VLA-style workflow:
`pretraining / supervised initialization → RL adaptation with simple task completion signal`

---

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

The episode return used by the optimizer is therefore:

$$
G_i^{\text{train}} = R_i
$$

This definition is used identically for `BASELINE`, `AC_LITE`, and `AC_FULL`. No reward shaping is applied.

---

## Environment Reward (Reporting Only)

`LunarLander-v2` internally produces dense environment rewards $r_t^{\text{env}}$. These are **not used for policy optimization**. They are retained only for diagnostics:

$$
G_i^{\text{env}} = \sum_{t} r_t^{\text{env}}
$$

In particular:
- $G_i^{\text{train}}$ is the reward used by PPO/GAE.
- $G_i^{\text{env}}$ is a raw environment performance statistic for analysis only.

---

## Noise Modes

| Mode           | Description                                   |
|----------------|-----------------------------------------------|
| `CLEAN`        | No corruption                                 |
| `REWARD_NOISE` | Terminal success labels are randomly flipped  |
| `OBS_NOISE`    | Gaussian noise is added to observations       |

### Reward Noise

With probability $p = 0.2$, a successful episode is converted into a failure label:

$$
R_i^{\text{policy}} = 0 \quad \text{even though} \quad R_i^{\text{raw}} = 1
$$

Earlier rewards remain unchanged. Reward noise directly corrupts the policy gradient under sparse reward.

### Observation Noise

$$
\tilde{s}_t = s_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2 I), \quad \sigma = 0.1
$$

Both policy and certainty networks receive corrupted observations.

---

## Methods

| Method     | Description                                                                         |
|------------|-------------------------------------------------------------------------------------|
| `BASELINE` | PPO-style policy optimization under sparse binary reward                            |
| `AC_LITE`  | Certainty-gated mixture policy objective; certainty trained by per-step mixture MLE |
| `AC_FULL`  | `AC_LITE` plus trajectory-level outcome MLE for certainty                           |

---

## Network Architecture

Policy and certainty networks are **fully independent**. No parameters are shared.

**Policy network:**
```
8 → 128 → 128 → 4
```

**Certainty network:**
```
8 → 128 → 128 → 1
```

- **Activation:** ReLU
- **Policy output:** $\pi_\theta(a \mid s)$ via softmax
- **Certainty output:** $u_t = f_\psi(s_t)$, $c_t = \sigma(u_t) \in (0, 1)$

Architectural isolation is necessary. A shared backbone would receive gradient signals from both the policy objective and the certainty objective simultaneously. Because these two objectives are generically non-aligned — and actively opposed when the policy is overconfident — sharing parameters creates an irresolvable tug-of-war. Independent networks guarantee that each objective trains its own parameters without interference.

---

## Pretrained Policy Backbone

Default path:
```text
pretrained_models/lunarlander_baseline_clean_seed42.pt
```

Two operating modes:
- **Fine-tuned policy:** Load pretrained weights and continue updating.
- **Frozen policy:** Load pretrained weights, keep policy fixed, train certainty only.

```python
freeze_pretrained_policy: bool
```

The pretrained checkpoint is treated as **checkpoint 0** and is a valid candidate for the best model.

---

## Baseline Policy Optimization

Standard PPO with Generalized Advantage Estimation.

**Default configuration:**

| Parameter           | Value              |
|---------------------|--------------------|
| Steps per update    | 2048               |
| Batch size          | 64                 |
| Optimizer           | Adam               |
| Learning rate       | $1 \times 10^{-4}$ |
| Discount $\gamma$   | 0.99               |
| GAE $\lambda$       | 0.95               |
| Entropy coefficient | 0.01               |
| Value coefficient   | 0.5                |
| Max gradient norm   | 0.5                |

GAE:

$$
\hat{A}_t = \text{GAE}(r_t^{\text{train}}, V(s_t), \gamma, \lambda)
$$

PPO objective:

$$
L_{\text{PPO}} = -\min\!\left( r_t(\theta)\,\hat{A}_t,\; \text{clip}\!\left(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon\right)\hat{A}_t \right)
$$

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}, \qquad \epsilon = 0.2
$$

---

## Grouped Rollouts and Dynamic Sampling

Episodes are collected in groups of `group_size = 4`.

Group filtering rule:
```python
keep_group = 0 < successes_in_group < group_size
```

Mixed-outcome groups are preferred. All-success and all-failure groups are uninformative for group-relative updates.

To prevent training starvation when mixed groups are temporarily rare:
```python
dynamic_sampling_fallback_on_empty = True
```

If no mixed groups are found in an update window, the full sample is used and the fallback event is logged.

---

## Rollout Temperature

$$
\pi_T(a \mid s) = \text{softmax}\!\left(\frac{z_\theta(s)}{T}\right), \qquad T = 1.0
$$

---

## AC-GRPO: Runner-Up Alignment and Certainty

### Runner-Up Action

At each timestep, the **runner-up action** is the highest-probability action the policy did not execute:

$$
\hat{a}_t = \arg\max_{a \neq a_t} \pi_\theta(a \mid s_t)
$$

### Runner-Up Margin

The alignment variable is the normalized margin between the executed action and its best competitor:

$$
\delta_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_\theta(a_t \mid s_t) + \pi_\theta(\hat{a}_t \mid s_t)} \in (0, 1)
$$

This quantity has the following properties:
- $\delta_t = 0.5$: the policy is indifferent between the executed action and its runner-up. This is the indifference point $c^\dagger = 0.5$.
- $\delta_t \to 1$: the policy is committed to the executed action.
- $\delta_t \to 0$: the runner-up dominates; the policy is actively preferring another action.
- At uniform initialization: $\pi_\theta(a_t) \approx \pi_\theta(\hat{a}_t) \approx 1/4$, so $\delta_t \approx 0.5$. Training begins exactly at the indifference point, with full dynamic range $[0, 1]$ available in both directions.

This is in contrast to the previous formulation $\delta_t = \pi_\theta(a_t|s_t)$, which creates a circular dependency: when the policy memorises a noisy reward label its probability for the executed action rises, $\delta_t \to 1$, certainty is trained toward $c_t \to 1$, and the policy gradient is maximally amplified on the noisy sample. The runner-up margin breaks this loop because a high $\pi_\theta(a_t)$ only produces a high $\delta_t$ when the policy simultaneously has a low $\pi_\theta(\hat{a}_t)$ — i.e., when it is genuinely committed and not merely memorising.

### Certainty

$$
c_t = \sigma\!\left(f_\psi(s_t)\right) \in (0, 1)
$$

The certainty network is trained to predict $\delta_t$ through the mixture MLE derived below. It does **not** receive gradients from the policy objective.

---

## Per-Step Mixture MLE

### Generative Model

At each step, we model the executed action as drawn from a mixture of two distributions: the policy's own distribution (which concentrates on $a_t$) and the runner-up distribution (which concentrates on $\hat{a}_t$), gated by certainty:

$$
p(a_t \mid s_t,\, c_t) = c_t \cdot \pi_\theta(a_t \mid s_t) + (1 - c_t) \cdot \pi_\theta(\hat{a}_t \mid s_t)
$$

The negative log-likelihood of this model is the **per-step mixture loss**:

$$
\boxed{\mathcal{L}_t^{\text{mix}} = -\log\!\left[ c_t \cdot \pi_\theta(a_t \mid s_t) + (1-c_t) \cdot \pi_\theta(\hat{a}_t \mid s_t) \right]}
$$

### Gradient on the Certainty Network

When $\pi_\theta$ is treated as fixed (stop-gradient), the gradient on $c_t$ is:

$$
\frac{\partial \mathcal{L}_t^{\text{mix}}}{\partial c_t}
= -\frac{\pi_\theta(a_t \mid s_t) - \pi_\theta(\hat{a}_t \mid s_t)}{c_t\,\pi_\theta(a_t \mid s_t) + (1-c_t)\,\pi_\theta(\hat{a}_t \mid s_t)}
$$

- When $\pi_\theta(a_t) > \pi_\theta(\hat{a}_t)$: gradient is negative → $c_t$ increases → certainty rises. ✓
- When $\pi_\theta(a_t) < \pi_\theta(\hat{a}_t)$: gradient is positive → $c_t$ decreases → certainty falls. ✓
- When $\pi_\theta(a_t) = \pi_\theta(\hat{a}_t)$: gradient is zero — exact indifference, $c^\dagger = 0.5$. ✓

### Gradient on the Policy Network

When $c_t$ is treated as fixed (stop-gradient), the gradient on $\theta$ is:

$$
\frac{\partial \mathcal{L}_t^{\text{mix}}}{\partial \theta}
= -\frac{c_t \cdot \nabla_\theta \pi_\theta(a_t \mid s_t) + (1-c_t) \cdot \nabla_\theta \pi_\theta(\hat{a}_t \mid s_t)}{c_t\,\pi_\theta(a_t \mid s_t) + (1-c_t)\,\pi_\theta(\hat{a}_t \mid s_t)}
$$

This is proportional to the standard policy gradient at initialization ($c_t \approx 0.5$, $\pi_\theta(a_t) \approx \pi_\theta(\hat{a}_t)$), ensuring that training begins immediately without a cold-start phase.

---

## Certainty-Gated Policy Objective

In the AC methods, the PPO probability ratio is computed using the mixture likelihood rather than the bare policy probability:

$$
q_t(\theta) = c_t^{\text{stop}} \cdot \pi_\theta(a_t \mid s_t) + (1-c_t^{\text{stop}}) \cdot \pi_\theta(\hat{a}_t \mid s_t)
$$

$$
r_t^{\text{AC}}(\theta) = \frac{q_t(\theta)}{q_t^{\text{old}}(\theta)}
$$

The PPO objective then becomes:

$$
L_{\text{PPO}}^{\text{AC}} = -\min\!\left( r_t^{\text{AC}}(\theta)\,\hat{A}_t,\; \text{clip}\!\left(r_t^{\text{AC}}(\theta),\, 1-\epsilon,\, 1+\epsilon\right)\hat{A}_t \right)
$$

The stop-gradient on $c_t$ prevents the policy loss from updating the certainty network. The advantage $\hat{A}_t$ is computed identically to the baseline via GAE on the sparse binary reward.

**Behavioural interpretation.** When $c_t \approx 1$, the mixture ratio reduces to the standard PPO ratio and the policy is updated at full strength. When $c_t \approx 0$, the executed action is effectively replaced by the runner-up in the ratio, reversing the gradient direction — the policy is pushed away from the executed action. When $c_t \approx 0.5$, the update is attenuated. This implements certainty-weighted policy optimization without any free scaling parameter.

---

## Trajectory-Level Outcome MLE (AC_FULL only)

At episode completion, the binary outcome $R_i \in \{0,1\}$ provides a direct signal about the reliability of the trajectory. The mean certainty over the trajectory is:

$$
\bar{c}_i = \frac{1}{T}\sum_{t=1}^T c_t
$$

We model the observed outcome as a Bernoulli draw gated by trajectory certainty:

$$
p(R_i \mid \bar{c}_i) = \bar{c}_i^{R_i}\,(1-\bar{c}_i)^{1-R_i}
$$

The negative log-likelihood is:

$$
\mathcal{L}_i^{\text{out}} = -R_i \log \bar{c}_i - (1-R_i)\log(1-\bar{c}_i)
$$

This trains the certainty network to predict episode success from within-episode state observations alone — without access to the true outcome at step time.

**Gradient on the certainty network.** For a successful episode ($R_i = 1$):

$$
\frac{\partial \mathcal{L}_i^{\text{out}}}{\partial c_t} = -\frac{1}{T \bar{c}_i} < 0
$$

The certainty is pushed upward for successful trajectories. For a failed episode ($R_i = 0$), the gradient is positive and certainty is pushed downward. Under reward noise, a flipped label ($R_i = 0$ for a true success) pushes certainty down on a trajectory where the per-step mixture MLE is simultaneously pushing certainty up (because the policy is committed to its actions). This **tension between the two MLE terms** is what allows the certainty to detect corrupted labels: the two sources of evidence disagree, and the certainty settles at an intermediate value rather than collapsing to 0 or 1.

**No free parameter.** The two MLE terms are log-likelihoods from independent data sources (per-step action observations and the terminal outcome) for the same latent variable $c_t$. Their sum is the joint MLE. No mixing coefficient is introduced.

---

## AC Method Variants

### AC_LITE

`AC_LITE` uses:
1. Certainty-gated mixture PPO for the policy network.
2. Per-step mixture MLE for the certainty network.

**Policy objective:**
$$
L_{\text{policy}}^{\text{AC-LITE}} = L_{\text{PPO}}^{\text{AC}}
$$

**Certainty objective:**
$$
\mathcal{L}_{\text{cert}}^{\text{AC-LITE}}
= -\frac{1}{T}\sum_{t=1}^T
\log\!\left[c_t \cdot \pi_\theta^{\text{stop}}(a_t \mid s_t)
+ (1-c_t) \cdot \pi_\theta^{\text{stop}}(\hat{a}_t \mid s_t)\right]
$$

`AC_LITE` asks a single question per step: *does the policy beat its own runner-up?* The certainty network learns to answer that question from observations alone, without access to any reward signal. This is the minimal viable AC mechanism.

### AC_FULL

`AC_FULL` uses:
1. Certainty-gated mixture PPO for the policy network.
2. Per-step mixture MLE for the certainty network.
3. Trajectory-level outcome MLE for the certainty network.

**Policy objective:**
$$
L_{\text{policy}}^{\text{AC-FULL}} = L_{\text{PPO}}^{\text{AC}}
$$

**Certainty objective (joint MLE over two independent data sources):**
$$
\mathcal{L}_{\text{cert}}^{\text{AC-FULL}}
= -\frac{1}{T}\sum_{t=1}^T
\log\!\left[c_t \cdot \pi_\theta^{\text{stop}}(a_t \mid s_t)
+ (1-c_t) \cdot \pi_\theta^{\text{stop}}(\hat{a}_t \mid s_t)\right]
- R_i \log \bar{c}_i - (1-R_i)\log(1-\bar{c}_i)
$$

`AC_FULL` adds the episode outcome as a second independent observation for certainty. Under reward noise, the two terms provide conflicting evidence for corrupted trajectories, which produces an intermediate certainty value and attenuates the policy gradient on those trajectories.

---

## Optimizer Configuration

Policy and certainty networks use **separate optimizers** with independent learning rates.

```python
policy_lr:    float  # default: 3e-4
certainty_lr: float  # default: 3e-4 (tune independently)
```

The gradient isolation between networks is guaranteed by architecture (no shared parameters) and enforced computationally by stop-gradients at the boundary of each network's input to the other's objective.

---

## Logged Quantities

**Episode-level:**
- Global step, episode ID
- Raw return $G_i^{\text{env}}$, policy outcome label $R_i^{\text{policy}}$, raw success $R_i^{\text{raw}}$
- Episode length

**Step-level:**
- Policy entropy $H_t = \mathcal{H}(\pi_\theta(\cdot|s_t))$
- Executed-action probability $\pi_\theta(a_t|s_t)$
- Runner-up probability $\pi_\theta(\hat{a}_t|s_t)$
- Runner-up margin $\delta_t$
- Certainty $c_t$
- Mixture probability $q_t$ (current policy likelihood in AC ratio)

**Update-level:**
- Policy loss, certainty loss (per-step and trajectory terms separately for AC_FULL)
- Gradient norm (policy and certainty separately)
- Mean $\delta_t$, mean $c_t$, mean $\bar{c}_i$
- Kept steps fraction, fallback flag

**Per-seed summary:**
- Method, mode, seed, total steps
- Full configuration, pretrained path, freeze flag
- Reward-noise semantics

---

## Evaluation Protocol

Evaluation uses a greedy policy.

All saved checkpoints are evaluated, including the pretrained anchor (checkpoint 0). The final checkpoint is not assumed to be optimal.

Held-out evaluation is mode-matched:
- `CLEAN` checkpoints evaluated on `CLEAN` held-out episodes.
- `OBS_NOISE` checkpoints evaluated on `OBS_NOISE` held-out episodes.
- `REWARD_NOISE` checkpoints evaluated on `REWARD_NOISE` held-out episodes.

**Primary metrics:**
- Success rate
- Raw environment return $G_i^{\text{env}}$
- Best checkpoint performance
- Seed mean and standard deviation

---

## Design Principle

The experiment isolates a single causal variable: **certainty-gated policy optimization**.

All methods share:
- Identical sparse binary reward
- Identical optimizer family
- Identical network architecture
- Identical rollout pipeline
- Identical pretrained anchor

Only the certainty mechanism differs. The AC objectives introduce no free hyperparameters beyond the learning rate: the per-step and trajectory-level terms are log-likelihoods from independent observations of the same latent certainty variable, and their combination follows from the joint MLE without a mixing coefficient.