# AC-PPO: Alignment-Certainty PPO Diagnostic

A `LunarLander-v2` diagnostic for **AC-PPO** — a PPO-based policy optimization variant that learns a per-timestep certainty signal to gate and stabilize policy updates.

The goal is to evaluate whether learned certainty can improve robustness when supervision is sparse, delayed, noisy, or dense but unreliable at the step level.

This diagnostic is designed as a controlled precursor to applying AC-style optimization to large-scale reinforcement learning systems such as **vision-language-action (VLA)** models.

The current repository implementation is **PPO-based**, not a standalone GRPO trainer. In other words:

- `BASELINE` is PPO with a critic and GAE.
- `AC_LITE` and `AC_FULL` are certainty-gated PPO variants built on the same actor-critic backbone.
- The pretrained anchor may come from an external or earlier **GRPO-style** baseline, but GRPO is **not** currently implemented here as a separate training branch.

At a high level, the methodology follows the same intuition as **SimpleVLA-RL**:

1. Start from a competent pretrained policy.
2. Perform RL adaptation under a simple reward signal — sparse binary outcome or dense per-step.
3. Evaluate all checkpoints explicitly; do not assume the final checkpoint is best.

> **Note:** This document defines the training protocol and mathematical formulation only. Experiment schedules, run commands, and empirical results are documented separately.

---

## Table of Contents

- [Task Setting](#task-setting)
- [Training Regime](#training-regime)
- [Reward Modes](#reward-modes)
- [Noise Modes](#noise-modes)
- [Methods](#methods)
- [Network Architecture](#network-architecture)
- [Pretrained Anchor](#pretrained-anchor)
- [Baseline Policy Optimization](#baseline-policy-optimization)
- [Grouped Rollouts and Dynamic Sampling](#grouped-rollouts-and-dynamic-sampling)
- [AC-PPO: Certainty and Alignment](#ac-ppo-certainty-and-alignment)
- [Per-Step Mixture MLE](#per-step-mixture-mle)
- [Certainty-Gated Policy Objective](#certainty-gated-policy-objective)
- [Trajectory-Level Outcome MLE — AC_FULL](#trajectory-level-outcome-mle--ac_full)
- [AC Method Variants](#ac-method-variants)
- [Optimizer Configuration](#optimizer-configuration)
- [Logged Quantities](#logged-quantities)
- [Evaluation Protocol](#evaluation-protocol)
- [Design Principle](#design-principle)

---

## Task Setting

| Quantity              | Value        |
|-----------------------|--------------|
| Environment           | `LunarLander-v2` |
| Observation dimension | 8            |
| Number of actions     | 4            |
| Max episode length    | 1000         |
| Success target        | Safe landing |

Each episode produces a binary outcome:

$$R_i \in \{0, 1\}$$

where `1` denotes successful landing and `0` denotes failure.

---

## Training Regime

All experiments are treated as **adaptation from a competent anchor policy**, not learning from scratch.

All compared methods start from the same pretrained checkpoint. The checkpoint may have been obtained under the original dense environment reward. That is acceptable because:

- The pretrained checkpoint serves only as a **common initialization**.
- All compared methods start from the same anchor.
- The adaptation phase tests robustness under the chosen reward and noise conditions.

This mirrors the intended VLA-style workflow:

```
pretraining / supervised initialization → RL adaptation with task completion signal
```

---

## Reward Modes

Two reward modes are supported. The reward mode is set per experiment and applies identically to `BASELINE`, `AC_LITE`, and `AC_FULL`.

### Sparse Reward

The training signal is a **terminal-only binary reward**.

$$r_t^{\text{train}} = 0 \quad \text{for } t < T$$

$$r_T^{\text{train}} = \begin{cases} 1 & \text{if success} \\ 0 & \text{if failure} \end{cases}$$

The episode return used by the optimizer is:

$$G_i^{\text{train}} = R_i$$

No reward shaping is applied.

Under sparse reward, `GAE` produces advantages that share the same sign for all steps within an episode — every step of a success gets a positive advantage, every step of a failure gets a negative advantage. The advantage therefore carries **episode-level reliability only**, with no within-episode differentiation. Certainty provides the only source of within-episode step weighting in this regime.

### Dense Reward

The training signal is the **per-step environment reward** produced natively by `LunarLander-v2`.

$$r_t^{\text{train}} = r_t^{\text{env}}$$

The episode return is:

$$G_i^{\text{train}} = \sum_t r_t^{\text{env}}$$

Under dense reward, `GAE` produces per-step advantages with genuine variation within an episode — good actions get positive advantages and bad actions get negative advantages. The advantage therefore already carries step-level information.

In this regime, certainty adds a second, **observation-driven** source of step-level weighting that is independent of the reward signal. A step may have a high advantage but low certainty (the policy hesitated before a lucky action) or low advantage but high certainty (the policy was committed but received a small reward). The product $c_t \cdot \hat{A}_t$ combines these two signals. Whether this combination improves on the advantage signal alone is an empirical question and one of the primary things this diagnostic is designed to test.

### Environment Reward (Reporting Only)

`LunarLander-v2` always produces dense environment rewards $r_t^{\text{env}}$ regardless of reward mode. These are retained for diagnostics in both modes:

$$G_i^{\text{env}} = \sum_t r_t^{\text{env}}$$

In sparse mode, $G_i^{\text{env}}$ is available for reporting but is **not** used for optimization.

---

## Noise Modes

| Mode           | Description                                   |
|----------------|-----------------------------------------------|
| `CLEAN`        | No corruption                                 |
| `REWARD_NOISE` | Terminal success labels are randomly flipped  |
| `OBS_NOISE`    | Gaussian noise added to observations          |

Noise modes are independent of reward mode. Any combination is valid.

### Reward Noise

With probability $p = 0.2$, a successful episode is converted to a failure label:

$$R_i^{\text{policy}} = 0 \quad \text{even though} \quad R_i^{\text{raw}} = 1$$

Under sparse reward, this directly corrupts the sign of the policy gradient for all steps in the affected episode. Under dense reward, reward noise corrupts the terminal bonus component but leaves the per-step shaping signal intact; the effect on the advantage is therefore smaller and less uniform.

Dynamic sampling and AC outcome supervision use the policy outcome label $R_i^{\text{policy}}$. The raw label $R_i^{\text{raw}}$ is retained for diagnostics only.

### Observation Noise

$$\tilde{s}_t = s_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2 I), \quad \sigma = 0.1$$

Both policy and certainty networks receive corrupted observations.

---

## Methods

| Method     | Description                                                                             |
|------------|-----------------------------------------------------------------------------------------|
| `BASELINE` | Standard PPO under the chosen reward signal, with GAE and a critic                      |
| `AC_LITE`  | PPO with detached certainty-gated advantages; certainty trained by per-step mixture MLE |
| `AC_FULL`  | `AC_LITE` plus trajectory-level outcome MLE for certainty (sparse reward only)          |

> **AC_FULL and dense reward.** The trajectory-level outcome MLE supervises certainty using the binary episode outcome $R_i$. Under dense reward, this binary signal is weaker relative to the per-step advantage signal, and the outcome NLL provides less marginal information over what the per-step mixture MLE already captures. `AC_FULL` is therefore most meaningful in the sparse reward setting where the outcome is the primary available signal. In dense reward experiments, `AC_LITE` is the recommended AC variant.

### Implemented Models in This Repository

The current codebase supports the following method families:

| Model family | Reward mode | Critic | Advantage estimator | Certainty gate |
|---|---|---:|---|---|
| PPO baseline | Sparse | Yes | GAE | No |
| PPO baseline | Dense | Yes | GAE | No |
| AC-PPO Lite | Sparse | Yes | GAE | Yes, detached $c_t \cdot \hat{A}_t$ |
| AC-PPO Lite | Dense | Yes | GAE | Yes, detached $c_t \cdot \hat{A}_t$ |
| AC-PPO Full | Sparse | Yes | GAE | Yes, detached $c_t \cdot \hat{A}_t$ |

`AC_FULL` is implemented in dense mode for compatibility, but its extra trajectory-level certainty loss is disabled there, so the dense `AC_FULL` path is effectively equivalent to dense `AC_LITE` in the current code.

### Pretrained Anchor

The default pretrained checkpoint may come from a GRPO-style clean baseline. In the current repository, however, that checkpoint is used only as a **shared initialization anchor** for PPO/AC-PPO adaptation. It is not presented as a separately trained comparison branch unless an explicit GRPO trainer is added later.

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
- **Certainty output:** $u_t = f_\psi(s_t)$, $\quad c_t = \sigma(u_t) \in (0, 1)$

**Why independence is required.** A shared backbone receives gradient signals from both the policy objective and the certainty objective simultaneously. These two objectives are generically non-aligned — and actively opposed when the policy is overconfident — so sharing parameters creates an irresolvable tug-of-war. Independent networks guarantee that each objective trains its own parameters without interference. See [gradient interference analysis](docs/theory.md) for the full derivation.

---

## Pretrained Anchor

Default path:
```
pretrained_models/lunarlander_baseline_clean_seed42.pt
```

Two operating modes:

- **Fine-tuned policy:** Load pretrained weights and continue updating $\theta$.
- **Frozen policy:** Load pretrained weights, fix $\theta$, train $\psi$ only.

```python
freeze_pretrained_policy: bool
```

The pretrained checkpoint is treated as **checkpoint 0** and is a valid candidate for the best model under the evaluation protocol.

---

## Baseline Policy Optimization

Standard PPO with Generalized Advantage Estimation.

**Default configuration:**

| Parameter           | Value              |
|---------------------|--------------------|
| Steps per update    | 2048               |
| Batch size          | 64                 |
| Optimizer           | Adam               |
| Learning rate       | $3 \times 10^{-4}$ |
| Discount $\gamma$   | 0.99               |
| GAE $\lambda$       | 0.95               |
| Entropy coefficient | 0.01               |
| Value coefficient   | 0.5                |
| Max gradient norm   | 0.5                |

**GAE:**

$$\hat{A}_t = \sum_{k=0}^{T-t} (\gamma\lambda)^k \left[ r_{t+k}^{\text{train}} + \gamma V(s_{t+k+1}) - V(s_{t+k}) \right]$$

**PPO objective:**

$$L_{\text{PPO}} = -\min\!\left( r_t(\theta)\,\hat{A}_t,\; \mathrm{clip}\!\left(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon\right)\hat{A}_t \right)$$

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}, \qquad \epsilon = 0.2$$

---

## Grouped Rollouts and Dynamic Sampling

Episodes are collected in groups of `group_size = 4`.

Mixed-outcome groups are preferred for group-relative updates:

```python
keep_group = 0 < successes_in_group < group_size
```

All-success and all-failure groups provide no contrastive signal and are discarded when possible.

To prevent training starvation when mixed groups are temporarily rare:

```python
dynamic_sampling_fallback_on_empty = True
```

If no mixed groups are found in an update window, the full sample is used and the fallback event is logged explicitly.

> Under dense reward, all-success and all-failure groups are defined by the binary episode outcome $R_i$, not by the dense return magnitude. The grouping logic is identical across reward modes.

---

## AC-PPO: Certainty and Alignment

### Runner-Up Action

At each timestep, the **runner-up action** is the highest-probability non-executed action:

$$\hat{a}_t = \arg\max_{a \neq a_t} \pi_\theta(a \mid s_t)$$

### Runner-Up Margin

The alignment variable $\delta_t$ is the normalised margin between the executed action and its best competitor:

$$\delta_t = \frac{\pi_\theta(a_t \mid s_t)}{\pi_\theta(a_t \mid s_t) + \pi_\theta(\hat{a}_t \mid s_t)} \in (0, 1)$$

Key properties:

| Value | Meaning |
|-------|---------|
| $\delta_t = 0.5$ | Policy is indifferent — indifference point $c^\dagger = 0.5$ |
| $\delta_t \to 1$ | Policy is committed to the executed action |
| $\delta_t \to 0$ | Runner-up dominates; policy prefers another action |
| Uniform init | $\pi(a_t) \approx \pi(\hat{a}_t) \approx 1/4$, so $\delta_t \approx 0.5$ |

Training begins exactly at the indifference point with full dynamic range $[0, 1]$ available in both directions.

> $\delta_t$ is a **diagnostic and training signal for the certainty network only**. It does not appear in the policy gradient. See [Certainty-Gated Policy Objective](#certainty-gated-policy-objective).

### Certainty

$$c_t = \sigma\!\left(f_\psi(s_t)\right) \in (0, 1)$$

The certainty network $f_\psi$ is trained to track $\delta_t$ through the mixture MLE derived below. It receives no gradients from the policy objective.

---

## Per-Step Mixture MLE

### Generative Model

At each step, the certainty network defines a latent gating model for the executed action:

$$p(a_t \mid s_t,\, c_t) = c_t \cdot \pi_\theta(a_t \mid s_t) + (1 - c_t) \cdot \pi_\theta(\hat{a}_t \mid s_t)$$

The interpretation: with probability $c_t$ the agent acted from genuine commitment (concentrated on $a_t$); with probability $1-c_t$ the agent was hesitant and the runner-up is the better representation of its intent.

The negative log-likelihood of this model is the **per-step mixture loss**:

$$\boxed{\mathcal{L}_t^{\text{mix}} = -\log\!\left[ c_t \cdot \pi_\theta(a_t \mid s_t) + (1-c_t) \cdot \pi_\theta(\hat{a}_t \mid s_t) \right]}$$

### Gradient on the Certainty Network

With $\pi_\theta$ fixed:

$$\frac{\partial \mathcal{L}_t^{\text{mix}}}{\partial c_t} = -\frac{\pi_\theta(a_t) - \pi_\theta(\hat{a}_t)}{c_t\,\pi_\theta(a_t) + (1-c_t)\,\pi_\theta(\hat{a}_t)}$$

| Condition | Effect |
|-----------|--------|
| $\pi_\theta(a_t) > \pi_\theta(\hat{a}_t)$ | $c_t$ increases — policy is committed |
| $\pi_\theta(a_t) < \pi_\theta(\hat{a}_t)$ | $c_t$ decreases — policy is hesitant |
| $\pi_\theta(a_t) = \pi_\theta(\hat{a}_t)$ | gradient zero — exact indifference |

The fixed point is $c_t^* = \delta_t$: certainty converges to the runner-up margin.

### Role of the Mixture Loss in the Current Code

In the current implementation, the mixture loss is used to train the **certainty network**, not to define the policy gradient. The actor update uses the standard PPO ratio, and the runner-up appears only inside the certainty objective and diagnostics.

The certainty loss itself is implemented without internal `.detach()` calls. Policy/certainty isolation is handled by:

- independent policy and certainty networks
- separate optimizers
- a detached certainty gate in the actor loss

---

## Certainty-Gated Policy Objective

Certainty enters the policy objective as a **scalar gate on the advantage**. The runner-up does not appear in the policy gradient.

$$\hat{A}_t^{\text{AC}} = c_t^{\text{stop}} \cdot \hat{A}_t$$

The PPO objective becomes:

$$L_{\text{PPO}}^{\text{AC}} = -\min\!\left( r_t(\theta)\,\hat{A}_t^{\text{AC}},\; \mathrm{clip}\!\left(r_t(\theta),\, 1-\epsilon,\, 1+\epsilon\right)\hat{A}_t^{\text{AC}} \right)$$

where $r_t(\theta)$ is the **standard PPO ratio**:

$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\text{old}}(a_t \mid s_t)}$$

The stop-gradient on $c_t$ prevents the policy loss from updating the certainty network.

In the current code this is implemented as:

```python
\hat{A}_t^{AC} = \mathrm{stopgrad}(c_t) \cdot \hat{A}_t
```

So the repository's AC methods are best understood as **certainty-gated PPO**, not as a separate runner-up-driven policy optimizer.

**Behavioural interpretation:**

| Certainty | Effect on policy update |
|-----------|------------------------|
| $c_t \approx 1$ | Full-strength update — step is reliable |
| $c_t \approx 0.5$ | Half-strength update — step is uncertain |
| $c_t \approx 0$ | Update suppressed — step is unreliable |

Low certainty suppresses the update. It does not reverse it or redirect it toward the runner-up.

**Dense reward note.** Under dense reward, $\hat{A}_t$ already varies per step. Certainty gating then combines two independent sources of step-level weighting: the reward-derived advantage and the observation-derived commitment signal. Steps with both high advantage and high certainty receive the strongest updates.

---

## Trajectory-Level Outcome MLE — AC_FULL

> Primarily intended for sparse reward. See [Methods](#methods) for the dense reward recommendation.

At episode completion, the binary outcome $R_i$ provides a direct signal about trajectory reliability. The mean certainty over the trajectory is:

$$\bar{c}_i = \frac{1}{T}\sum_{t=1}^T c_t$$

We model the observed outcome as a Bernoulli observation gated by trajectory-mean certainty:

$$p(R_i \mid \bar{c}_i) = \bar{c}_i^{R_i}\,(1-\bar{c}_i)^{1-R_i}$$

The negative log-likelihood:

$$\mathcal{L}_i^{\text{out}} = -R_i \log \bar{c}_i - (1-R_i)\log(1-\bar{c}_i)$$

**Noise robustness mechanism.** For a corrupted success ($R_i^{\text{policy}} = 0$, $R_i^{\text{raw}} = 1$):

- The per-step mixture loss pushes $c_t$ **up** — the policy was committed to its actions.
- The outcome NLL pushes $c_t$ **down** — the label says the episode failed.

These two MLE terms provide conflicting evidence for corrupted trajectories. Certainty settles at an intermediate value, and $c_t \cdot \hat{A}_t$ attenuates the corrupted policy gradient. No explicit noise detection is required.

**No free parameter.** The per-step and trajectory terms are log-likelihoods from independent data sources — per-step action observations and the terminal outcome — for the same latent variable $c_t$. Their sum is the joint MLE. No mixing coefficient is introduced or needed.

---

## AC Method Variants

### AC_LITE

**Policy objective:** Standard PPO with certainty-gated advantages.

**Certainty objective:** Per-step mixture MLE.

$$\mathcal{L}_{\text{cert}}^{\text{AC-LITE}} = -\frac{1}{T}\sum_{t=1}^T \log\!\left[c_t \cdot \pi_\theta(a_t \mid s_t) + (1-c_t) \cdot \pi_\theta(\hat{a}_t \mid s_t)\right]$$

`AC_LITE` asks one question per step: *is the policy more committed to the executed action than to its best alternative?* The certainty network learns to answer from observations alone, with no access to reward. This is the minimal viable AC mechanism and the recommended variant for dense reward experiments.

### AC_FULL

**Policy objective:** Standard PPO with certainty-gated advantages (identical to AC_LITE).

**Certainty objective:** Joint MLE over per-step and trajectory-level observations.

$$\mathcal{L}_{\text{cert}}^{\text{AC-FULL}} = \underbrace{-\frac{1}{T}\sum_{t=1}^T \log\!\left[c_t \cdot \pi_\theta(a_t \mid s_t) + (1-c_t) \cdot \pi_\theta(\hat{a}_t \mid s_t)\right]}_{\text{per-step mixture MLE}} \underbrace{- R_i \log \bar{c}_i - (1-R_i)\log(1-\bar{c}_i)}_{\text{trajectory outcome MLE}}$$

`AC_FULL` adds the episode outcome as a second independent observation for certainty. The two terms can conflict on corrupted trajectories, producing the noise-detection behaviour described above. Primarily meaningful under sparse reward with reward noise.

---

## Optimizer Configuration

Policy and certainty networks use **separate optimizers** with independent learning rates.

```python
policy_lr:    float  # current default: 1e-4
certainty_lr: float  # current default: 1e-4
```

Gradient isolation is guaranteed by architecture (no shared parameters) and enforced by the training loop (separate `zero_grad / backward / step` blocks). The actor loss uses a detached certainty gate, while the certainty losses themselves contain no internal `.detach()` calls.

### Current Reproduction-Critical Defaults

The current implementation uses the following important defaults:

| Parameter | Value |
|---|---:|
| `policy_lr` | `1e-4` |
| `certainty_lr` | `1e-4` |
| `update_epochs` | `4` |
| `checkpoint_interval` | `10_000` |
| `eval_seeds` | `(101, 102, 103)` |
| `eval_episodes_per_seed` | `5` |
| `group_size` | `4` |
| `max_group_attempts_per_update` | `256` |
| `dynamic_sampling_warmup_steps` | `10_000` in the current experiment grid |
| `skip_policy_update_on_unmixed_fallback` | `True` |

Pretrained loading is actor-only by default unless critic loading is explicitly enabled in code.

---

## Logged Quantities

### Episode-level
| Quantity | Description |
|----------|-------------|
| `global_step` | Total environment steps |
| `episode_id` | Episode index |
| `return_env` | $G_i^{\text{env}}$ — raw dense return (all modes) |
| `return_train` | $G_i^{\text{train}}$ — return used for optimization |
| `outcome_policy` | $R_i^{\text{policy}}$ — label seen by optimizer |
| `outcome_raw` | $R_i^{\text{raw}}$ — true outcome before noise |
| `episode_length` | $T$ |

### Step-level
| Quantity | Description |
|----------|-------------|
| `policy_entropy` | $H_t = \mathcal{H}(\pi_\theta(\cdot\mid s_t))$ |
| `action_prob` | $\pi_\theta(a_t \mid s_t)$ |
| `runner_up_prob` | $\pi_\theta(\hat{a}_t \mid s_t)$ |
| `delta` | Runner-up margin $\delta_t$ |
| `certainty` | $c_t$ |

### Update-level
| Quantity | Description |
|----------|-------------|
| `policy_loss` | PPO loss value |
| `cert_loss_step` | Per-step mixture NLL |
| `cert_loss_traj` | Trajectory outcome NLL (AC_FULL only) |
| `grad_norm_theta` | Policy network gradient norm |
| `grad_norm_psi` | Certainty network gradient norm |
| `delta_mean` / `delta_min` / `delta_max` | Runner-up margin statistics |
| `certainty_mean` | Mean $c_t$ over update batch |
| `c_bar_mean` | Mean trajectory certainty $\bar{c}_i$ (AC_FULL) |
| `kept_steps_frac` | Fraction of rollout steps used after group filtering |
| `fallback_used` | Whether dynamic sampling fallback was triggered |

---

## Evaluation Protocol

Evaluation uses a **greedy policy** ($T = 0$, argmax actions).

All saved checkpoints are evaluated, including the pretrained anchor (checkpoint 0). The final checkpoint is not assumed to be optimal.

Held-out evaluation is **mode-matched**:

| Training mode | Evaluation episodes |
|---------------|---------------------|
| `CLEAN` | Clean held-out episodes |
| `OBS_NOISE` | Observation-noisy held-out episodes |
| `REWARD_NOISE` | Clean held-out episodes (reward noise is a training-only corruption) |

**Primary metrics:**
- Success rate (binary landing outcome)
- Raw environment return $G_i^{\text{env}}$
- Best checkpoint performance across all seeds
- Seed mean ± standard deviation

---

## Design Principle

The experiment isolates a single causal variable: **certainty-gated policy optimization**.

All methods share:

- Identical reward signal (sparse or dense, set per experiment)
- Identical noise conditions
- Identical optimizer family and hyperparameters
- Identical network architecture
- Identical rollout pipeline
- Identical pretrained anchor

Only the certainty mechanism differs. In implementation terms, the current project compares **PPO** against **certainty-gated PPO** on the same actor-critic backbone.

The core theoretical claim being tested is:

> A certainty signal trained from action commitment alone — with no direct access to reward — can identify unreliable training steps and reduce their contribution to the PPO policy gradient, improving robustness under noisy or sparse supervision.
