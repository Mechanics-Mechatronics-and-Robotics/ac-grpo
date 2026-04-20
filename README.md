# AC-GRPO Methodology

This repository implements a `LunarLander-v2` diagnostic for **Alignment-Certainty Group Relative Policy Optimization (AC-GRPO)**. The goal is to study whether a learned per-timestep certainty signal can make policy optimization more robust when supervision is noisy or unreliable.

> **Note:** This document describes the method and training pipeline only. Experiment plans, run commands, and empirical results are intentionally kept out of this README and should be documented separately.

## Task Setting

The diagnostic environment is `LunarLander-v2` with a discrete action space.

| Quantity | Value |
|---|---:|
| Observation dimension | 8 |
| Number of actions | 4 |
| Max episode length | 1000 |
| Success target | Safe landing |

Each episode produces a binary outcome:

$$
R_i \in \{0, 1\}
$$

where `1` denotes a successful landing and `0` denotes failure.

The pipeline supports three environment modes:

| Mode | Description |
|---|---|
| `CLEAN` | No corruption. |
| `REWARD_NOISE` | False-negative success corruption. Successful episodes may be mislabelled as failures for policy learning. |
| `OBS_NOISE` | Gaussian observation noise is added to observations. |

For `REWARD_NOISE`, corruption is applied to the reward signal used by PPO/GAE rather than only to auxiliary logging labels. This is critical because the policy must experience the same noisy supervision that the method is designed to handle. In implementation, when a successful episode is flipped to a failure signal, the terminal rollout reward used for advantage estimation is modified before GAE computation.

## Methods

The code compares three methods:

| Method | Description |
|---|---|
| `BASELINE` | PPO-style policy optimization with a repaired sampling and clipping pipeline. |
| `AC_LITE` | Baseline policy optimization plus certainty-gated advantages and alignment loss. |
| `AC_FULL` | `AC_LITE` plus outcome loss and a discrete-action dispersion proxy. |

The AC variants are intentionally small extensions of the baseline pipeline. The certainty head does not select actions and shares no parameters with the policy.

## Networks

The policy and certainty functions are implemented as independent MLPs.

**Policy network:**
```
8 → 128 → 128 → 4
```

**Certainty network:**
```
8 → 128 → 128 → 1
```

Both use ReLU hidden activations. The policy produces a categorical action distribution over four discrete actions. The certainty network produces a scalar logit:

$$
u_t = f_\psi(s_t)
$$

and certainty is computed as:

$$
c_t = \sigma(u_t)
$$

The implementation clamps certainty to the range:

$$
c_t \in [10^{-6}, 1 - 10^{-6}]
$$

The policy and certainty networks are strictly isolated:

$$
s_t \rightarrow \pi_\theta(a \mid s_t) \quad \text{and} \quad s_t \rightarrow c_\psi(s_t)
$$

No parameters are shared. This isolation prevents a degenerate solution where the policy could reduce its own training signal by jointly lowering action quality and certainty.

## Pretrained Policy Backbone

The current pipeline supports loading a pretrained policy checkpoint as the starting backbone for fine-tuning.

The default pretrained policy path is:
```text
pretrained_models/lunarlander_baseline_clean_seed42.pt
```

This checkpoint is loaded into `PolicyNet` when `pretrained_policy_path` is specified in the config. The policy can be used in two ways:

| Mode | Behavior |
|---|---|
| Fine-tuned policy | Load pretrained weights and continue updating policy parameters. |
| Frozen policy | Load pretrained weights and freeze policy parameters while training auxiliary AC components. |

The frozen-policy option is controlled by:
```python
freeze_pretrained_policy: bool
```

This option is useful for separating two research questions:
1. Whether the certainty model can learn meaningful reliability signals on top of a fixed policy.
2. Whether certainty-gated policy optimization improves the policy when gradients are allowed to update the policy.

## Baseline Policy Optimization

The baseline is a compact PPO-style trainer using the following defaults:

| Hyperparameter | Value |
|---|---:|
| Steps per update | 2048 |
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate | $3 \times 10^{-4}$ |
| Discount ($\gamma$) | 0.99 |
| GAE $\lambda$ | 0.95 |
| Entropy coefficient | 0.01 |
| Value coefficient | 0.5 |
| Max gradient norm | 0.5 |

The baseline uses Generalized Advantage Estimation (GAE):

$$
\hat{A}_t = \text{GAE}(r_t, V_\theta(s_t), \gamma, \lambda)
$$

and PPO clipping:

$$
L_{\text{PPO}} = -\min\left(
    r_t(\theta)\hat{A}_t,
    \text{clip}\big(r_t(\theta), 1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}\big)\hat{A}_t
\right)
$$

where the probability ratio is:

$$
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

The implementation supports asymmetric clipping:
```python
epsilon_low = 0.2
epsilon_high = 0.2
```

The same code path can support `clip-higher` variants by increasing `epsilon_high`, but the default methodology uses symmetric clipping unless explicitly changed.

## Grouped Rollouts and Dynamic Sampling

The repaired baseline pipeline supports grouped rollouts. A group is a set of complete episodes collected under the current policy. The default group size is:

```python
group_size = 4
```

Dynamic sampling uses binary outcomes within each group:
- All-failure groups are considered uninformative for group-relative updates.
- All-success groups are considered uninformative for group-relative updates.
- Mixed-outcome groups are preferred because they contain within-policy contrast.

The intended filtering rule is:
```python
keep_group = 0 < successes_in_group < group_size
```

However, early in training mixed groups may be rare. To prevent complete training starvation, the implementation includes a fallback:
```python
dynamic_sampling_fallback_on_empty = True
```

When no mixed groups are found in an update window, the trainer falls back to the sampled groups for that update and logs the fallback event. This keeps optimization moving while still measuring how often true mixed-outcome sampling is available.

The pipeline logs:
- Fraction of discarded groups
- Fraction of mixed-outcome groups
- Mean successes per group
- Whether fallback was used
- Policy entropy
- Gradient norm

## Rollout Temperature

Action sampling can use a rollout temperature:

$$
\pi_T(a \mid s) = \text{softmax}\left(\frac{z_\theta(s)}{T}\right)
$$

where $T = 1.0$ is the default. Higher temperatures increase exploration by flattening the action distribution during rollout collection.

The current recommended setting uses:
```python
rollout_temperature = 1.0
```

## AC-GRPO Certainty Variables

For each sampled transition:

$$
\delta_t = \pi_\theta(a_t \mid s_t)
$$

where $a_t$ is the action actually taken.

The certainty head produces:

$$
c_t = \sigma(u_t), \quad u_t = f_\psi(s_t)
$$

For discrete `LunarLander` actions, the executed-action probability is bounded by $\delta_t \in [0, 1]$, and under a uniform four-action policy $\delta_t \approx 0.25$. Thus, certainty values collapsing near zero are not expected to be useful for the policy gradient. To prevent certainty from completely shutting off learning, AC policy gating uses an effective gate:

$$
c_t^{\text{eff}} = c_t(1 - c_{\min}) + c_{\min}
$$

with $c_{\min} = 0.3$.

The policy advantage is gated by $c_t^{\text{eff}}$, not raw certainty:

$$
\hat{A}^{\text{AC}}_t = \text{stopgrad}(c_t^{\text{eff}}) \hat{A}_t
$$

The stop-gradient is critical: the policy loss must not update the certainty network through the gate.

## AC Alignment Loss

The alignment loss trains certainty to match the policy's own commitment to the sampled action:

$$
L_{\text{align}}(t) = -\delta_t \log c_t - (1-\delta_t)\log\frac{1-c_t}{K}
$$

For `LunarLander`, $K = 4$.

The implementation detaches $\delta_t$ so this loss trains the certainty network rather than pushing policy probabilities through the certainty objective.

The alignment loss also supports a certainty temperature:

$$
c_{t,T} = \sigma\left(\frac{\text{logit}(c_t)}{T}\right)
$$

Then $c_{t,T}$ is used in the alignment objective. The default is:
```python
ac_loss_temperature = 1.0
```

Temperature values $> 1.0$ soften certainty targets; values $< 1.0$ sharpen them.

## AC Outcome Loss

`AC_FULL` adds an episode-outcome likelihood. For each timestep belonging to a completed episode:

$$
L_{\text{outcome}}(t) = -\alpha \left[ R_i \log c_t + (1-R_i)\log(1-c_t) \right]
$$

with $\alpha = 1.0$.

This couples local certainty to trajectory-level success. The implementation masks incomplete rollout fragments so the outcome loss is applied only to timesteps whose episode outcome is known.

## AC Dispersion Proxy for Discrete Actions

The original AC-GRPO orbit term is motivated by continuous action geometry. `LunarLander` uses discrete actions, so the implementation uses policy entropy as a discrete dispersion surrogate:

$$
H_t = \mathcal{H}\big(\pi_\theta(\cdot \mid s_t)\big)
$$

The dispersion proxy is:

$$
L_{\text{dispersion}}(t) = \frac{1}{2}\exp(u_t)H_t - \frac{1}{2}\beta u_t
$$

with $\beta = 1.0$.

This is not the exact continuous Gaussian orbit likelihood. It is a discrete-action surrogate that encourages low certainty when the policy distribution is diffuse and higher certainty when the policy is concentrated.

Policy entropy is detached in this loss so the dispersion proxy trains the certainty network without pushing policy logits through the certainty objective.

## AC Model Variants

### AC_LITE

`AC_LITE` uses:
1. PPO policy loss with certainty-gated advantages
2. Alignment loss for the certainty network

The certainty objective is:
$$
\mathcal{L}_{\text{cert}}^{\text{AC-LITE}} = \mathcal{L}_{\text{align}}
$$

### AC_FULL

`AC_FULL` uses:
1. PPO policy loss with certainty-gated advantages
2. Alignment loss
3. Outcome loss
4. Entropy-based dispersion proxy

The certainty objective is:
$$
\mathcal{L}_{\text{cert}}^{\text{AC-FULL}} = \mathcal{L}_{\text{align}} + \mathcal{L}_{\text{outcome}} + \mathcal{L}_{\text{dispersion}}
$$

The policy and certainty optimizers are separate. Certainty-derived gates are detached in the policy loss, and policy-derived targets are detached in certainty losses where appropriate.

## Noise Models

### Observation Noise

In `OBS_NOISE`, observations are corrupted with Gaussian noise:

$$
\tilde{s}_t = s_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2 I)
$$

with $\sigma = 0.1$. Both the policy and certainty networks receive the corrupted observation.

### Reward Noise

In `REWARD_NOISE`, the intended corruption is false-negative success supervision. A true successful landing can be converted into a failure signal with probability $p = 0.2$.

The key implementation detail is that the corrupted outcome affects the reward used by PPO/GAE, not merely an auxiliary logging label. This makes the noisy-supervision setting fair for comparing methods that are supposed to handle unreliable rewards.

## Reproducibility and Logged Quantities

Each training run records per-episode and per-step CSV logs.

**Episode-level logs include:**
- Global step
- Episode ID
- Return
- Logged success
- Raw success
- Episode length

**Step-level logs include:**
- Global step
- Episode ID
- Timestep
- Policy entropy
- Executed-action probability $\delta_t$
- Certainty $c_t$

**Baseline update logs also include:**
- Discarded group fraction
- Mixed group fraction
- Mean successes per group
- Mean policy entropy
- Number of kept steps
- Dynamic-sampling fallback flag
- Loss
- Gradient norm

**Per-seed JSON summaries include:**
- Method
- Mode
- Seed
- Total steps
- Full training config
- Runtime metadata
- Pretrained policy path
- Whether the pretrained policy was frozen
- Reward-noise semantics
- Certainty-gate semantics (for AC runs)

Generated experiment reports summarize seed aggregation and AUROC diagnostics. Reports are saved as `report.md` at the experiment folder root. Plots are saved in the `plots/` subfolder.