# README.md

# AC-GRPO Methodology (Sparse Reward)

This repository implements a `LunarLander-v2` diagnostic for **Alignment-Certainty Group Relative Policy Optimization (AC-GRPO)** under a **sparse terminal reward** regime.

The goal is to evaluate whether a learned per-timestep certainty signal can stabilize policy optimization and improve robustness when supervision is sparse, delayed, or unreliable.

This diagnostic is designed as a controlled precursor to applying AC-style optimization to large-scale reinforcement learning systems such as **vision-language-action (VLA)** models.

At a high level, the methodology follows the same intuition as **SimpleVLA-RL**:

1. start from a competent pretrained policy,
2. perform RL adaptation with a simple binary outcome signal,
3. evaluate checkpoints explicitly instead of assuming the final checkpoint is best.

Note:
This document defines the training protocol and mathematical formulation only.
Experiment schedules, run commands, and empirical results must be documented separately.

---

# Task Setting

The diagnostic environment is `LunarLander-v2` with a discrete action space.

| Quantity              |        Value |
| --------------------- | -----------: |
| Observation dimension |            8 |
| Number of actions     |            4 |
| Max episode length    |         1000 |
| Success target        | Safe landing |

Each episode produces a binary outcome:

R_i ∈ {0, 1}

where:

* `1` — successful landing
* `0` — failure

This binary outcome is the only learning signal used for policy optimization.

---

# Training Regime and Common Anchor

The sparse-reward experiments are treated as **adaptation from a competent anchor policy**, not sparse-reward learning from scratch.

In the current methodology, all compared methods may start from the same pretrained policy checkpoint. This pretrained checkpoint may itself have been obtained under the original dense environment reward. That is acceptable because:

* the pretrained checkpoint serves only as a **common initialization**,
* all compared methods start from the same anchor,
* the sparse-reward phase tests **adaptation under binary outcome supervision** rather than scratch learning.

This mirrors the intended VLA-style workflow:

pretraining or supervised initialization → RL adaptation with simple task completion signal.

---

# Sparse Reward Definition

The training signal is a **terminal-only binary reward**.

At each timestep:

r_t^train = 0                      for t < T
r_T^train = 1                      if success
r_T^train = 0                      if failure

Therefore the episode return used by the optimizer is:

G_i^train = R_i

This definition is used identically for:

* BASELINE
* AC_LITE
* AC_FULL

No reward shaping is applied.
No dense reward is used for learning.

This ensures strict comparability between methods and aligns the toy setup with outcome-level RL intuition.

---

# Environment Reward (Reporting Only)

`LunarLander-v2` internally produces dense environment rewards:

r_t^env

These values are **not used for policy optimization**.

They are retained only for reporting and diagnostics:

G_i^env = Σ_t r_t^env

This separation enables:

* consistent sparse learning,
* interpretable performance metrics,
* stable comparison across methods,
* compatibility with binary-outcome RL pipelines.

In particular:

* `G_i^train` is the reward used by PPO / GAE,
* `G_i^env` is a raw environment performance statistic for analysis only.

---

# Noise Modes

The pipeline supports three supervision conditions.

| Mode         | Description                                  |
| ------------ | -------------------------------------------- |
| CLEAN        | No corruption                                |
| REWARD_NOISE | Terminal success labels are randomly flipped |
| OBS_NOISE    | Gaussian noise is added to observations      |

These modes simulate realistic deployment failures in robotics and VLA systems.

---

# Reward Noise (Sparse Regime)

Reward corruption applies only to terminal outcomes.

With probability:

p = 0.2

a successful episode is converted into a failure label:

R_i^policy = 0
even though
R_i^raw = 1

Implementation:

if success and random() < p:
r_T_train = 0

Earlier rewards remain:

r_t_train = 0    for t < T

Thus reward noise directly affects the policy gradient in the sparse-reward regime.

Dynamic sampling and AC outcome supervision use the policy outcome label:

R_i^policy

while raw environment return remains available for diagnostics.

---

# Observation Noise

Observation corruption is applied at every timestep:

s_t_tilde = s_t + ε_t

where:

ε_t ~ Normal(0, σ² I)

Default:

sigma = 0.1

Both the policy and certainty networks receive corrupted observations.

---

# Methods

The repository compares three training pipelines.

| Method   | Description                                                                     |
| -------- | ------------------------------------------------------------------------------- |
| BASELINE | PPO-style policy optimization under sparse binary reward                        |
| AC_LITE  | Baseline policy optimization plus certainty-gated advantages and alignment loss |
| AC_FULL  | AC_LITE plus outcome loss and a discrete-action dispersion proxy                |

The AC extensions are intentionally small.
The policy architecture remains unchanged.

---

# Network Architecture

Policy and certainty networks are fully independent.

Policy network:

8 → 128 → 128 → 4

Certainty network:

8 → 128 → 128 → 1

Activation:

ReLU

Policy output:

π_θ(a | s)

Certainty output:

u_t = f_ψ(s_t)

c_t = sigmoid(u_t)

Certainty is clamped:

c_t ∈ [1e-6, 1 - 1e-6]

No parameters are shared between policy and certainty networks.

This isolation prevents degenerate feedback loops where the policy suppresses its own learning signal by manipulating certainty.

---

# Pretrained Policy Backbone

Training may start from a pretrained policy checkpoint.

Default:

pretrained_models/lunarlander_baseline_clean_seed42.pt

Two operating modes are supported.

**Fine-tuned policy**

Load pretrained weights and continue updating parameters.

**Frozen policy**

Load pretrained weights and keep policy fixed while training certainty.

Configuration:

freeze_pretrained_policy: bool

This separation allows controlled investigation of two research questions:

1. Can certainty learn meaningful reliability signals on top of a fixed policy?
2. Can certainty-gated optimization improve the policy itself?

The pretrained checkpoint is always considered part of the evaluation trajectory.
It should be treated as **checkpoint 0**, and it is allowed to be the best model.

---

# Baseline Policy Optimization

The baseline uses PPO with Generalized Advantage Estimation.

Default trainer configuration:

Steps per update: 2048
Batch size: 64
Optimizer: Adam
Learning rate: 3e-4
Discount γ: 0.99
GAE λ: 0.95
Entropy coefficient: 0.01
Value coefficient: 0.5
Max gradient norm: 0.5

For sparse-reward adaptation from a pretrained checkpoint, a smaller learning rate may be used in practice. Such changes belong to the experiment configuration, not to the method definition.

GAE:

A_t = GAE(r_t^train, V(s_t), γ, λ)

PPO objective:

L_PPO =

* min(
  r_t(θ) * A_t,
  clip(r_t(θ), 1 - ε_low, 1 + ε_high) * A_t
  )

Probability ratio:

r_t(θ) =
π_θ(a_t | s_t)
/
π_old(a_t | s_t)

Clipping defaults:

epsilon_low  = 0.2
epsilon_high = 0.2

The code path also supports asymmetric clipping variants, but symmetric clipping is the default methodology unless explicitly overridden.

---

# Grouped Rollouts and Dynamic Sampling

Episodes are collected in groups.

Default:

group_size = 4

Dynamic sampling uses binary outcomes within each group.

Group filtering rule:

keep_group =
0 < successes_in_group < group_size

So:

* all-failure groups are uninformative for group-relative updates,
* all-success groups are uninformative for group-relative updates,
* mixed-outcome groups are preferred.

However, especially in a toy task, mixed groups may be temporarily rare. To prevent total training starvation, the implementation supports:

dynamic_sampling_fallback_on_empty = True

If no mixed groups are found in an update window, the sampled groups are used for that update and the fallback event is logged explicitly.

Logged statistics include:

* discarded group fraction,
* mixed group fraction,
* mean successes per group,
* fallback usage,
* policy entropy,
* gradient norm.

This behavior is intentionally close to outcome-driven RL practice while remaining robust enough for a toy diagnostic.

---

# Rollout Temperature

Sampling temperature modifies exploration:

π_T(a | s) =
softmax(z_θ(s) / T)

Default:

rollout_temperature = 1.0

Higher temperatures increase exploration by flattening the rollout policy.

---

# AC-GRPO Certainty Variables

Executed-action probability:

δ_t =
π_θ(a_t | s_t)

Certainty:

c_t =
sigmoid(u_t)

Effective certainty gate:

c_eff =
c_t * (1 - c_min) + c_min

Default:

c_min = 0.3

Gated advantage:

A_t_AC =
stopgrad(c_eff) * A_t

The stop-gradient prevents the policy loss from directly updating the certainty network through the gate.

This keeps the certainty signal auxiliary and avoids degenerate coupling.

---

# Alignment Loss

The alignment loss trains certainty to match the policy's own commitment to the executed action.

L_align =

* δ_t * log(c_t)
* (1 - δ_t) * log((1 - c_t) / K)

For LunarLander:

K = 4

The probability δ_t is detached so this loss trains the certainty network only.

This makes alignment a self-consistency objective rather than an additional policy objective.

---

# Outcome Loss (AC_FULL)

AC_FULL adds a trajectory-level Bernoulli likelihood:

L_outcome =

* α [
  R_i * log(c_t)
  +
  (1 - R_i) * log(1 - c_t)
  ]

Default:

alpha = 1.0

The outcome loss is applied only after episode completion, when the trajectory outcome is known.

---

# Dispersion Proxy (Discrete Actions)

The original AC-GRPO orbit term is motivated by continuous action geometry.

`LunarLander-v2` has discrete actions, so AC_FULL uses policy entropy as a discrete dispersion surrogate.

Policy entropy:

H_t =
Entropy(π_θ(. | s_t))

Dispersion loss:

L_dispersion =
0.5 * exp(u_t) * H_t

* 0.5 * β * u_t

Default:

beta = 1.0

Policy entropy is detached so the loss trains the certainty network only.

This is not the exact continuous Gaussian orbit likelihood.
It is a discrete-action surrogate suitable for the toy setting.

---

# AC Model Variants

## AC_LITE

AC_LITE uses:

1. PPO policy loss with certainty-gated advantages
2. Alignment loss for the certainty network

Certainty objective:

L_cert^AC-LITE = L_align

## AC_FULL

AC_FULL uses:

1. PPO policy loss with certainty-gated advantages
2. Alignment loss
3. Outcome loss
4. Entropy-based dispersion proxy

Certainty objective:

L_cert^AC-FULL = L_align + L_outcome + L_dispersion

The policy and certainty optimizers are separate.

Certainty-derived gates are detached in the policy loss.
Policy-derived quantities are detached in certainty losses where appropriate.

---

# Logged Quantities

Each run records per-episode and per-step logs.

Episode-level logs include:

* global step,
* episode id,
* raw return,
* policy outcome label,
* raw success,
* episode length.

Step-level logs include:

* timestep,
* policy entropy,
* executed-action probability δ_t,
* certainty c_t.

Update-level logs include:

* loss,
* gradient norm,
* mean policy entropy,
* kept steps,
* fallback flag.

Per-seed summaries include:

* method,
* mode,
* seed,
* total steps,
* full configuration,
* pretrained policy path,
* frozen policy flag,
* reward-noise semantics,
* certainty-gate parameters.

---

# Evaluation Protocol

Evaluation always uses a greedy policy.

Checkpoint selection is performed by evaluating **all saved checkpoints**, including the pretrained anchor checkpoint, on fixed held-out episodes.

The final checkpoint is not assumed to be optimal.

Held-out evaluation is mode-matched:

* CLEAN checkpoints are evaluated on CLEAN held-out episodes,
* OBS_NOISE checkpoints are evaluated on OBS_NOISE held-out episodes,
* REWARD_NOISE checkpoints are evaluated on REWARD_NOISE held-out episodes.

Primary metrics:

* success rate,
* raw environment return,
* best checkpoint performance,
* seed mean and standard deviation.

This protocol is intentionally checkpoint-centric, since outcome-driven RL can degrade after early gains.

---

# Design Principle

The experiment isolates a single causal variable:

certainty-gated policy optimization

All methods share:

* identical sparse binary reward,
* identical optimizer family,
* identical architecture,
* identical rollout pipeline,
* identical pretrained anchor when used.

Only the certainty mechanism differs.

This ensures that any observed performance difference can be attributed to certainty-aware optimization rather than reward engineering.