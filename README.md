# AC-GRPO Methodology (Sparse Reward)

This repository implements a `LunarLander-v2` diagnostic for **Alignment-Certainty Group Relative Policy Optimization (AC-GRPO)** under a sparse terminal reward regime.

The goal is to evaluate whether a learned per-timestep certainty signal can stabilize policy optimization and improve robustness when supervision is sparse, delayed, or unreliable.

This diagnostic is designed as a controlled precursor to applying AC-style optimization to large-scale reinforcement learning systems such as vision-language-action (VLA) models.

Note:  
This document defines the training protocol and mathematical formulation only.  
Experiment schedules, run commands, and empirical results must be documented separately.

---

# Task Setting

The diagnostic environment is `LunarLander-v2` with a discrete action space.

| Quantity | Value |
|----------|------:|
| Observation dimension | 8 |
| Number of actions | 4 |
| Max episode length | 1000 |
| Success target | Safe landing |

Each episode produces a binary outcome:

R_i ∈ {0, 1}

where:

1 — successful landing  
0 — failure  

This binary outcome is the only learning signal used for policy optimization.

---

# Sparse Reward Definition

The training signal is a terminal-only binary reward.

At each timestep:

r_t^train = 0                      for t < T  
r_T^train = 1                      if success  
r_T^train = 0                      if failure  

Therefore the episode return used by the optimizer is:

G_i^train = R_i

This definition is used identically for:

- BASELINE  
- AC_LITE  
- AC_FULL  

No reward shaping is applied.  
No dense reward is used for learning.  

This ensures strict comparability between methods.

---

# Environment Reward (Reporting Only)

`LunarLander-v2` internally produces dense environment rewards:

r_t^env

These values are not used for policy optimization.

They are retained only for reporting and diagnostics:

G_i^env = Σ_t r_t^env

This separation enables:

- consistent sparse learning
- interpretable performance metrics
- stable comparison across methods
- direct compatibility with binary-reward RL pipelines

---

# Noise Modes

The pipeline supports three supervision conditions.

| Mode | Description |
|------|------------|
| CLEAN | No corruption |
| REWARD_NOISE | Terminal success labels randomly flipped |
| OBS_NOISE | Gaussian noise added to observations |

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

This ensures reward noise directly affects the policy gradient.

---

# Observation Noise

Observation corruption is applied at every timestep:

s_t_tilde = s_t + ε_t

where:

ε_t ~ Normal(0, σ² I)

Default:

sigma = 0.1

Both policy and certainty networks receive corrupted observations.

---

# Methods

The repository compares three training pipelines.

| Method | Description |
|--------|------------|
| BASELINE | PPO-style policy optimization |
| AC_LITE | Certainty-gated advantages plus alignment loss |
| AC_FULL | AC_LITE plus outcome and dispersion losses |

The AC extensions are intentionally minimal.

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

This isolation prevents degenerate feedback loops where the policy suppresses its own learning signal.

---

# Pretrained Policy Backbone

Training may start from a pretrained policy checkpoint.

Default:

pretrained_models/lunarlander_baseline_clean_seed42.pt

Two operating modes are supported.

Fine-tuned policy:

Load pretrained weights and continue updating parameters.

Frozen policy:

Load pretrained weights and keep policy fixed while training certainty.

Configuration:

freeze_pretrained_policy: bool

This separation allows controlled investigation of two research questions:

1) Can certainty learn meaningful reliability signals on top of a fixed policy?  
2) Can certainty-gated optimization improve the policy itself?

---

# Baseline Policy Optimization

The baseline uses PPO with Generalized Advantage Estimation.

Default configuration:

Steps per update: 2048  
Batch size: 64  
Optimizer: Adam  
Learning rate: 3e-4  
Discount γ: 0.99  
GAE λ: 0.95  
Entropy coefficient: 0.01  
Value coefficient: 0.5  
Max gradient norm: 0.5  

GAE:

A_t = GAE(r_t^train, V(s_t), γ, λ)

PPO objective:

L_PPO =
- min(
    r_t(θ) * A_t,
    clip(r_t(θ), 1 - ε_low, 1 + ε_high) * A_t
)

Probability ratio:

r_t(θ) =
π_θ(a_t | s_t)
/
π_old(a_t | s_t)

Clipping:

epsilon_low  = 0.2  
epsilon_high = 0.2  

---

# Grouped Rollouts and Dynamic Sampling

Episodes are collected in groups.

Default:

group_size = 4

Group filtering rule:

keep_group =
0 < successes_in_group < group_size

Fallback mechanism:

dynamic_sampling_fallback_on_empty = True

If no mixed groups are found in an update window, sampled groups are used to avoid training starvation.

Logged statistics:

- discarded group fraction
- mixed group fraction
- successes per group
- fallback usage
- policy entropy
- gradient norm

---

# Rollout Temperature

Sampling temperature modifies exploration.

π_T(a | s) =
softmax(z_θ(s) / T)

Default:

rollout_temperature = 1.0

Higher temperature increases exploration.

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

The stop-gradient prevents the policy from directly updating the certainty network.

---

# Alignment Loss

The alignment loss trains certainty to match the policy's commitment to the executed action.

L_align =
- δ_t * log(c_t)
- (1 - δ_t) * log((1 - c_t) / K)

For LunarLander:

K = 4

The probability δ_t is detached so the loss trains the certainty network only.

---

# Outcome Loss (AC_FULL)

AC_FULL adds a trajectory-level likelihood.

L_outcome =
- α [
    R_i * log(c_t)
    +
    (1 - R_i) * log(1 - c_t)
]

Default:

alpha = 1.0

The loss is applied only after episode completion.

---

# Dispersion Proxy (Discrete Actions)

The dispersion term approximates continuous orbit regularization.

Policy entropy:

H_t =
Entropy(π_θ(. | s_t))

Dispersion loss:

L_dispersion =
0.5 * exp(u_t) * H_t
- 0.5 * β * u_t

Default:

beta = 1.0

Policy entropy is detached so the loss trains certainty only.

---

# Logged Quantities

Episode-level logs:

- global step
- episode id
- raw return
- success label
- episode length

Step-level logs:

- timestep
- policy entropy
- executed-action probability
- certainty

Update-level logs:

- loss
- gradient norm
- entropy
- kept steps
- fallback flag

Per-seed summaries include:

- method
- mode
- seed
- total steps
- full configuration
- pretrained policy path
- frozen policy flag
- reward-noise configuration
- certainty gate parameters

---

# Evaluation Protocol

Evaluation always uses:

- greedy policy  
- no reward noise  
- no observation noise  

Metrics:

- success rate  
- episode return  
- best checkpoint performance  
- seed mean and standard deviation  

The final checkpoint is not assumed to be optimal.

All saved checkpoints are evaluated.

The best-performing checkpoint is selected using held-out evaluation episodes.

---

# Design Principle

The experiment isolates a single causal variable:

certainty-gated policy optimization

All methods share:

- identical sparse reward
- identical optimizer
- identical architecture
- identical rollout pipeline

Only the certainty mechanism differs.

This ensures that any observed performance difference can be attributed to certainty-aware optimization rather than reward engineering.