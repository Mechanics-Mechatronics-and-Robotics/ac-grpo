# AC-GRPO: Alignment-Certainty Group Relative Policy Optimization

## Core Idea

Standard GRPO assigns uniform credit to every action token within a
trajectory based solely on its outcome reward. A token executed near
a contact singularity receives the same gradient signal as one in
open space. A policy that fails on the final step receives the same
zero reward as one that fails on the first. Certainty is invisible
to the optimizer.

AC-GRPO replaces uniform token credit with a **learned, per-timestep
certainty gate** $c_t \in (0,1)$ that modulates gradient contribution
based on the policy's self-assessed reliability at each state. This
gate is not a heuristic weight. It is derived from a joint maximum
likelihood objective over three independent observations of the same
latent certainty variable — making AC-GRPO a strict probabilistic
generalization of GRPO that reduces to it when $c_t = 1$ everywhere.

---

## Generative Model

For each timestep $t$ in trajectory $\tau_i$, the certainty variable
$c_t = \sigma(r_\psi(s_t))$ is posited to explain three independent
observations:

1. **Alignment** — how confidently the policy committed to the action it took
2. **Action orbit** — how spread the policy's action distribution is at $s_t$
3. **Trajectory outcome** — whether the trajectory succeeded or failed

The joint negative log-likelihood over all three is the AC-GRPO loss.

---

## Three Likelihood Terms

### Term 1: Bernoulli Alignment

Define the per-timestep alignment variable as the probability the
policy assigned to the executed action:

$$\delta_t = \pi_\theta(a_t \mid s_t) \in [0,1]$$

Under the Partitioned Categorical model with $K$ alternative actions,
the alignment likelihood is the AC-loss applied per token:

$$\mathcal{L}_{\text{align}}^{(t)} =
-\delta_t \log c_t
-(1-\delta_t)\log\frac{1-c_t}{K}$$

Gradient with respect to certainty logit $u_t = \log\frac{c_t}{1-c_t}$:

$$\frac{\partial \mathcal{L}_{\text{align}}^{(t)}}{\partial u_t} = c_t - \delta_t$$

Fixed point: $c_t^* = \delta_t$.
The certainty head learns to mirror the policy's own commitment —
a **self-consistency constraint** that requires no external label.

---

### Term 2: Action Orbit Geometry

GRPO samples $G$ trajectories per state, providing $G$ action samples
$\{a_t^{(i)}\}_{i=1}^G$ at each $s_t$. Define the action orbit
displacement as the mean pairwise difference:

$$\|\Delta a_t\|^2 = \frac{1}{G(G-1)}\sum_{i \neq j}\|a_t^{(i)} - a_t^{(j)}\|^2$$

Model this displacement as a zero-mean isotropic Gaussian whose
precision is parameterized by the certainty logit:

$$\Delta a_t \mid u_t \sim \mathcal{N}(0,\ e^{-u_t} I_d)$$

The negative log-likelihood is:

$$\mathcal{L}_{\text{orbit}}^{(t)} =
\frac{e^{u_t}}{2}\|\Delta a_t\|^2 - \frac{d}{2}u_t$$

MLE fixed point:

$$c_t^* = \sigma\!\left(\log\frac{d}{\|\Delta a_t\|^2}\right)$$

This term connects certainty to action geometry directly:

| Action distribution | $\|\Delta a_t\|^2$ | $c_t^*$ |
|---|---|---|
| Peaked (confident) | Small | $\to 1$ |
| Diffuse (uncertain) | Large | $\to 0$ |
| Near-contact / cluttered | Large | $\to 0$ |

Crucially, this term provides **non-zero gradients to the certainty
head even when all trajectories fail** — the orbit geometry is
computable regardless of outcome reward.

---

### Term 3: Trajectory Outcome Coupling

The binary outcome $R_i \in \{0,1\}$ couples the per-timestep
certainty to trajectory-level success via a Bernoulli likelihood:

$$P(R_i \mid c_t) = c_t^{\alpha R_i}(1-c_t)^{\alpha(1-R_i)}$$

The negative log-likelihood per token:

$$\mathcal{L}_{\text{outcome}}^{(t)} =
-\alpha\left[R_i \log c_t + (1-R_i)\log(1-c_t)\right]$$

This drives certainty high throughout successful trajectories and
low throughout failed ones, coupling local reliability estimates to
global task performance.

---

## Unified MLE Objective

The three terms share the same latent variable $c_t$.
No weighting hyperparameter. No combined loss.
A single maximum likelihood objective:

$$\boxed{
\mathcal{L}_{\text{AC-GRPO}}^{(t)} =
\underbrace{
    -\delta_t \log c_t
    -(1-\delta_t)\log\frac{1-c_t}{K}
}_{\text{alignment}}
+
\underbrace{
    \frac{e^{u_t}}{2}\|\Delta a_t\|^2
    -\frac{d}{2}u_t
}_{\text{orbit}}
+
\underbrace{
    -\alpha\left[R_i \log c_t + (1-R_i)\log(1-c_t)\right]
}_{\text{outcome}}
}$$

The full gradient with respect to certainty logit $u_t$:

$$\frac{\partial \mathcal{L}_{\text{AC-GRPO}}^{(t)}}{\partial u_t} =
\underbrace{(c_t - \delta_t)}_{\text{self-consistency residual}}
+\underbrace{\frac{e^{u_t}}{2}\|\Delta a_t\|^2 - \frac{d}{2}}_{\text{orbit residual}}
-\underbrace{\alpha(R_i - c_t)}_{\text{outcome residual}}$$

MLE fixed point for certainty:

$$c_t^* = \frac{\delta_t + \alpha R_i}{1+\alpha}
\quad \text{(modulated by orbit geometry)}$$

Certainty is the weighted average of **what the policy committed to**
($\delta_t$) and **what the trajectory achieved** ($R_i$), stabilized
by the geometric spread of the action distribution.

---

## Certainty-Gated Policy Gradient

Standard GRPO advantage:

$$\hat{A}_i =
\frac{R_i - \text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^G)}$$

AC-GRPO replaces this with a **per-token, certainty-weighted advantage**:

$$\hat{A}_{i,t}^{\text{AC}} =
c_{i,t} \cdot
\frac{R_i - \bar{R}_c}{\text{std}_c(\{R_j\})}$$

where the certainty-weighted group mean is:

$$\bar{R}_c =
\frac{\displaystyle\sum_{j=1}^G c_{j,t}\, R_j}
{\displaystyle\sum_{j=1}^G c_{j,t}}$$

The policy gradient becomes:

$$\nabla_\theta \mathcal{J}_{\text{AC-GRPO}} =
\frac{1}{G}\sum_{i=1}^G \frac{1}{|\tau_i|}\sum_{t=1}^{|\tau_i|}
c_{i,t} \cdot \hat{A}_{i,t}^{\text{AC}} \cdot
\nabla_\theta \log \pi_\theta(a_{i,t} \mid s_{i,t})$$

The certainty gate $c_{i,t}$ plays three simultaneous roles:

| Role | Mechanism |
|---|---|
| **Token-level credit** | Downweights uncertain timesteps regardless of outcome |
| **Baseline modulation** | Uncertain trajectories contribute less to the group mean |
| **Adaptive curriculum** | Bottleneck states suppressed until policy is ready |

---

## Architecture: Isolation Principle

Following the AC-loss design philosophy, prediction and certainty
are computed by **fully independent parameter sets**:

s_t  →  Policy Backbone    (θ)  →  π_θ(a|s_t)   [prediction]

s_t  →  Certainty Backbone (ψ)  →  c_t           [certainty]

Isolation prevents degenerate solutions in which the policy suppresses
its own loss by jointly degrading actions and certainty through shared
parameters. The certainty backbone cannot influence action selection,
and the policy gradient does not flow through the certainty parameters.

---

## Resolution of GRPO Failure Modes

### Failure Mode 1: Zero-reward collapse

**Standard GRPO**: when all $G$ trajectories fail, $\hat{A}_i = 0$
for all $i$ — null gradient, training stalls.

**AC-GRPO**: the orbit term and alignment term continue training the
certainty head from geometry alone, independent of $R_i$. When the
first successful trajectory appears, the certainty gate is already
calibrated. The policy gradient immediately amplifies reliable
timesteps and suppresses unreliable ones.

---

### Failure Mode 2: Uniform token credit

**Standard GRPO**: every token in a successful trajectory receives
identical gradient signal, including lucky actions in ambiguous states.

**AC-GRPO**: the per-token gate $c_{i,t}$ weights each token by its
certified reliability. Near-contact states, cluttered configurations,
and visually ambiguous scenes receive low $c_{i,t}$ and contribute
weakly to the policy update regardless of trajectory outcome.

---

### Failure Mode 3: Low success rate threshold

**Standard GRPO**: dynamic sampling requires mixed-outcome groups —
infeasible when the success rate is near zero.

**AC-GRPO**: the orbit term provides a stable geometric training
signal at zero success rate. The certainty head identifies bottleneck
states — those with large $\|\Delta a_t\|^2$ — and suppresses
gradient there, concentrating policy improvement on states where the
policy is already near-certain. This geometric curriculum operates
**below** the success rate threshold required by standard GRPO.

---

## Relationship to AC-Loss Family

| Loss | Setting | Certainty target |
|---|---|---|
| `AC-loss` | Supervised classification | Label alignment $\delta = p_y / \|p\|$ |
| `ACC-loss` | Self-supervised (SimCLR) | Contrastive alignment $\delta = p_{i,i^+}$ |
| `AC-GRPO` | Reinforcement learning (GRPO) | Action alignment $\delta = \pi_\theta(a_t \mid s_t)$ |

All three are instances of the same generative model: a Partitioned
Categorical likelihood over a scalar certainty variable, extended by
a Gaussian orbit likelihood that grounds certainty in the geometric
structure of the prediction space. The RL setting adds a third
Bernoulli likelihood coupling per-timestep certainty to trajectory
outcome, making the fixed point a weighted average of local commitment
and global success.
