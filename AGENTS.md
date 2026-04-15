
# AGENTS.md

## Project

AC-GRPO sanity check on LunarLander-v2.

Goal: verify in one day whether certainty-gated policy optimization behaves sensibly under noisy supervision.

This is a diagnostic experiment, not a benchmark.

---

## Core Question

Does a learned per-timestep certainty signal improve robustness when reward or observation signals are unreliable?

---

## Time Budget

Approximate allocation:

- 2h baseline
- 2h AC-lite
- 2h noise + debugging
- 2h 5-seed runs
- 1h plots + summary

Do not expand scope.

---

## Environment

Gym: LunarLander-v2  
Discrete actions: K = 4  
Observation size: 8  
Max episode length: 1000  

Define:

success = safe landing  

Binary outcome:

R_i ∈ {0,1}

---

## Experimental Modes

### CLEAN

No corruption.

### REWARD_NOISE

False-negative corruption only.

Implementation:

if R_i == 1 and random() < p:
    R_i = 0

p = 0.2

Rationale:

Models unreliable success detection rather than artificial success.

### OBS_NOISE

Gaussian observation noise.

sigma = 0.1

obs = obs + Normal(0, sigma)

---

## Methods

Implement exactly three.

1. BASELINE  
   Standard PPO / GRPO-style trainer.

2. AC_LITE  
   Certainty gate + alignment loss.

3. AC_FULL  
   Alignment + outcome + dispersion proxy.

---

## Networks

Policy:

MLP  
8 → 128 → 128 → 4  

Certainty:

MLP  
8 → 128 → 128 → 1  

Activation: ReLU  

No parameter sharing.

Isolation principle enforced.

---

## Certainty Variables

delta_t = pi_theta(a_t | s_t)

c_t = sigmoid(u_t)

Clamp:

c_t = clamp(c_t, 1e-6, 1-1e-6)

Important note:

For discrete actions:

delta_t ∈ [1/K, 1]

Therefore certainty near zero should be rare.

Monitor histogram.

---

## Losses

### Alignment

L_align(t) =
    - delta_t log c_t
    - (1 - delta_t) log((1 - c_t) / K)

K = 4

---

### Outcome

Binary success target.

L_outcome(t) =
    - alpha [
        R_i log c_t
        + (1 - R_i) log(1 - c_t)
    ]

alpha = 1.0

---

### Dispersion Proxy (Discrete Action Adaptation)

Original orbit term assumes continuous action geometry.

Discrete LunarLander requires adaptation.

We use policy entropy as a dispersion proxy.

Let:

H_t = entropy(pi_theta(. | s_t))

Define:

D_t = H_t

Then:

L_dispersion(t) =
    0.5 exp(u_t) D_t
    - 0.5 beta u_t

beta = 1.0

Important:

This is not the exact Gaussian orbit likelihood.  
It is a discrete-action surrogate.

Document this explicitly in reports.

---

## Policy Update

Certainty-gated advantage.

A_ac(i,t) =
    stopgrad(c(i,t)) * A(i,t)

Critical rule:

certainty must be detached.

No gradient from policy loss into certainty network.

---

## Training

Steps per update:

2048

Batch size:

64

Optimizer:

Adam

Learning rate:

3e-4

Gamma:

0.99

Lambda:

0.95

Clip:

0.2

Total steps:

200k

---

## Seeds

Use exactly:

42  
0  
17  
9  
3  

Debug:

seed 42 only.

---

## Metrics

### Episode-Level

return  
success  
episode length  

### Step-Level

certainty  
policy entropy  
delta_t  

---

## AUROC Metrics

### Trajectory-Level

Mean certainty predicts final success.

mean_cert_per_traj = mean(c_t)

AUROC_traj =
    roc_auc(success, mean_cert_per_traj)

Primary paper metric.

---

### Timestep-Level

Certainty predicts difficult phase.

Define:

late_phase = t > 0.8 * episode_length

AUROC_step =
    roc_auc(late_phase, 1 - certainty)

Diagnostic metric.

---

## Required Plots

Generate exactly:

1. Return vs steps
2. Success rate vs steps
3. Certainty histogram
4. Certainty vs entropy scatter
5. Certainty vs delta_t scatter
6. Clean vs noisy return comparison

---

## Acceptance Criteria

Valid experiment requires:

1. baseline learns in CLEAN
2. reward noise degrades baseline
3. AC_LITE trains stably
4. certainty varies across states
5. runs reproducible across seeds

If AC == baseline:

hypothesis not supported.

That is acceptable.

---

## Failure Checks

Monitor:

NaN loss  
certainty collapse to 0  
certainty collapse to 1  
zero gradients  
no learning progress  

If failure:

reduce learning rate  
disable dispersion term  
inspect certainty histogram  

---

## Repository Structure

repo/

AGENTS.md  
README.md  
requirements.txt  

src/

env.py  
policy_net.py  
certainty_net.py  
trainer_baseline.py  
trainer_ac.py  
losses.py  
metrics.py  

scripts/

train_baseline.py  
train_ac.py  
run_all_seeds.py  
analyze.py  

outputs/

logs  
plots  
checkpoints  

---

## Minimal Run Plan

Execute strictly in order.

1. Train baseline CLEAN
2. Add certainty logging
3. Add certainty gating
4. Add alignment loss
5. Add reward noise
6. Add observation noise
7. Add outcome loss
8. Add dispersion proxy
9. Run 5 seeds
10. Generate plots

Do not reorder.

---

## Expected Outcome

Reasonable result:

baseline performance drops under noise  
AC degrades less  
certainty correlates with instability  

If not observed:

the method likely provides no benefit in this regime.

Report honestly.

---

## End
