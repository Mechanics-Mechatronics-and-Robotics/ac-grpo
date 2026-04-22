# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-22_231909_all_experiments`

Reproducibility files: `config.yaml`, `summary.json`, per-seed `*_summary.json`, and per-seed CSV logs are generated with each run. Git tracks only `report.md` by default; generated logs/checkpoints/plots are ignored.

## Experiment protocol

- Implemented method variants in this run: AC_FULL_SPARSE, AC_LITE_DENSE, AC_LITE_SPARSE, BASELINE_DENSE, BASELINE_SPARSE
- Training modes: CLEAN, OBS_NOISE, REWARD_NOISE
- Reward modes represented: DENSE, SPARSE
- Training seeds: 0, 3, 9, 17, 42
- All branches start from the shared pretrained anchor when a pretrained path is provided.
- Training uses grouped rollouts with dynamic sampling fallback when no mixed-outcome groups are available.
- Checkpoints are saved during training and the pretrained anchor is treated as checkpoint 0.
- Checkpoint selection uses greedy held-out evaluation under the branch's primary selection condition.
- Additional challenge tests evaluate selected checkpoints under CLEAN, OBS_NOISE (typical), and OBS_NOISE (hard).
- Challenge tests currently use up to 2620 episodes per evaluation seed in the generated logs.

## Summary table (mean ± std over seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | 268.6 ± 17.7 | 0.940 ± 0.089 | 288.1 | 1.000 |
| CLEAN | AC_LITE_DENSE | 265.8 ± 11.1 | 0.930 ± 0.057 | 285.8 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 263.6 ± 17.3 | 0.910 ± 0.055 | 286.9 | 1.000 |
| CLEAN | BASELINE_DENSE | 263.0 ± 14.6 | 0.920 ± 0.045 | 286.4 | 0.990 |
| CLEAN | BASELINE_SPARSE | 256.4 ± 10.5 | 0.910 ± 0.042 | 284.0 | 1.000 |
| OBS_NOISE | AC_FULL_SPARSE | 164.8 ± 29.6 | 0.520 ± 0.115 | 198.6 | 0.680 |
| OBS_NOISE | AC_LITE_DENSE | 211.6 ± 34.1 | 0.710 ± 0.139 | 249.7 | 0.860 |
| OBS_NOISE | AC_LITE_SPARSE | 117.1 ± 41.8 | 0.320 ± 0.175 | 199.3 | 0.670 |
| OBS_NOISE | BASELINE_DENSE | 188.0 ± 27.5 | 0.600 ± 0.100 | 244.4 | 0.840 |
| OBS_NOISE | BASELINE_SPARSE | 150.8 ± 38.4 | 0.460 ± 0.164 | 198.0 | 0.660 |
| REWARD_NOISE | AC_FULL_SPARSE | 232.8 ± 37.3 | 0.640 ± 0.147 | 279.7 | 0.840 |
| REWARD_NOISE | AC_LITE_DENSE | 262.9 ± 13.1 | 0.720 ± 0.045 | 286.4 | 0.910 |
| REWARD_NOISE | AC_LITE_SPARSE | 248.0 ± 24.2 | 0.670 ± 0.115 | 282.8 | 0.860 |
| REWARD_NOISE | BASELINE_DENSE | 266.1 ± 15.2 | 0.720 ± 0.144 | 284.8 | 0.900 |
| REWARD_NOISE | BASELINE_SPARSE | 255.2 ± 26.0 | 0.710 ± 0.119 | 280.7 | 0.860 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | 0 | 280.5 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 3 | 254.7 | 0.900 |
| CLEAN | AC_FULL_SPARSE | 9 | 280.9 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 17 | 282.5 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 42 | 244.6 | 0.800 |
| CLEAN | AC_LITE_DENSE | 0 | 273.7 | 0.950 |
| CLEAN | AC_LITE_DENSE | 3 | 255.6 | 0.900 |
| CLEAN | AC_LITE_DENSE | 9 | 254.6 | 0.850 |
| CLEAN | AC_LITE_DENSE | 17 | 265.1 | 0.950 |
| CLEAN | AC_LITE_DENSE | 42 | 280.0 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 0 | 267.7 | 0.850 |
| CLEAN | AC_LITE_SPARSE | 3 | 233.0 | 0.850 |
| CLEAN | AC_LITE_SPARSE | 9 | 272.0 | 0.950 |
| CLEAN | AC_LITE_SPARSE | 17 | 274.3 | 0.950 |
| CLEAN | AC_LITE_SPARSE | 42 | 270.8 | 0.950 |
| CLEAN | BASELINE_DENSE | 0 | 257.6 | 0.900 |
| CLEAN | BASELINE_DENSE | 3 | 271.6 | 0.950 |
| CLEAN | BASELINE_DENSE | 9 | 276.4 | 0.950 |
| CLEAN | BASELINE_DENSE | 17 | 240.1 | 0.850 |
| CLEAN | BASELINE_DENSE | 42 | 269.3 | 0.950 |
| CLEAN | BASELINE_SPARSE | 0 | 271.7 | 0.950 |
| CLEAN | BASELINE_SPARSE | 3 | 256.6 | 0.900 |
| CLEAN | BASELINE_SPARSE | 9 | 255.7 | 0.900 |
| CLEAN | BASELINE_SPARSE | 17 | 256.0 | 0.950 |
| CLEAN | BASELINE_SPARSE | 42 | 242.2 | 0.850 |
| OBS_NOISE | AC_FULL_SPARSE | 0 | 169.6 | 0.500 |
| OBS_NOISE | AC_FULL_SPARSE | 3 | 171.4 | 0.550 |
| OBS_NOISE | AC_FULL_SPARSE | 9 | 124.2 | 0.400 |
| OBS_NOISE | AC_FULL_SPARSE | 17 | 205.4 | 0.700 |
| OBS_NOISE | AC_FULL_SPARSE | 42 | 153.3 | 0.450 |
| OBS_NOISE | AC_LITE_DENSE | 0 | 168.0 | 0.550 |
| OBS_NOISE | AC_LITE_DENSE | 3 | 192.9 | 0.600 |
| OBS_NOISE | AC_LITE_DENSE | 9 | 251.1 | 0.850 |
| OBS_NOISE | AC_LITE_DENSE | 17 | 206.3 | 0.700 |
| OBS_NOISE | AC_LITE_DENSE | 42 | 239.8 | 0.850 |
| OBS_NOISE | AC_LITE_SPARSE | 0 | 117.7 | 0.300 |
| OBS_NOISE | AC_LITE_SPARSE | 3 | 80.8 | 0.150 |
| OBS_NOISE | AC_LITE_SPARSE | 9 | 69.8 | 0.150 |
| OBS_NOISE | AC_LITE_SPARSE | 17 | 158.5 | 0.500 |
| OBS_NOISE | AC_LITE_SPARSE | 42 | 158.6 | 0.500 |
| OBS_NOISE | BASELINE_DENSE | 0 | 182.7 | 0.550 |
| OBS_NOISE | BASELINE_DENSE | 3 | 215.9 | 0.700 |
| OBS_NOISE | BASELINE_DENSE | 9 | 143.9 | 0.450 |
| OBS_NOISE | BASELINE_DENSE | 17 | 203.2 | 0.650 |
| OBS_NOISE | BASELINE_DENSE | 42 | 194.2 | 0.650 |
| OBS_NOISE | BASELINE_SPARSE | 0 | 190.1 | 0.650 |
| OBS_NOISE | BASELINE_SPARSE | 3 | 96.8 | 0.250 |
| OBS_NOISE | BASELINE_SPARSE | 9 | 184.5 | 0.600 |
| OBS_NOISE | BASELINE_SPARSE | 17 | 148.8 | 0.400 |
| OBS_NOISE | BASELINE_SPARSE | 42 | 133.6 | 0.400 |
| REWARD_NOISE | AC_FULL_SPARSE | 0 | 281.5 | 0.700 |
| REWARD_NOISE | AC_FULL_SPARSE | 3 | 242.8 | 0.850 |
| REWARD_NOISE | AC_FULL_SPARSE | 9 | 187.6 | 0.500 |
| REWARD_NOISE | AC_FULL_SPARSE | 17 | 204.4 | 0.500 |
| REWARD_NOISE | AC_FULL_SPARSE | 42 | 248.0 | 0.650 |
| REWARD_NOISE | AC_LITE_DENSE | 0 | 277.0 | 0.650 |
| REWARD_NOISE | AC_LITE_DENSE | 3 | 242.8 | 0.750 |
| REWARD_NOISE | AC_LITE_DENSE | 9 | 258.0 | 0.700 |
| REWARD_NOISE | AC_LITE_DENSE | 17 | 266.4 | 0.750 |
| REWARD_NOISE | AC_LITE_DENSE | 42 | 270.1 | 0.750 |
| REWARD_NOISE | AC_LITE_SPARSE | 0 | 268.5 | 0.650 |
| REWARD_NOISE | AC_LITE_SPARSE | 3 | 275.0 | 0.800 |
| REWARD_NOISE | AC_LITE_SPARSE | 9 | 246.6 | 0.750 |
| REWARD_NOISE | AC_LITE_SPARSE | 17 | 233.3 | 0.650 |
| REWARD_NOISE | AC_LITE_SPARSE | 42 | 216.8 | 0.500 |
| REWARD_NOISE | BASELINE_DENSE | 0 | 263.5 | 0.650 |
| REWARD_NOISE | BASELINE_DENSE | 3 | 241.9 | 0.500 |
| REWARD_NOISE | BASELINE_DENSE | 9 | 270.6 | 0.850 |
| REWARD_NOISE | BASELINE_DENSE | 17 | 272.2 | 0.800 |
| REWARD_NOISE | BASELINE_DENSE | 42 | 282.4 | 0.800 |
| REWARD_NOISE | BASELINE_SPARSE | 0 | 272.5 | 0.700 |
| REWARD_NOISE | BASELINE_SPARSE | 3 | 273.8 | 0.800 |
| REWARD_NOISE | BASELINE_SPARSE | 9 | 239.9 | 0.550 |
| REWARD_NOISE | BASELINE_SPARSE | 17 | 273.5 | 0.850 |
| REWARD_NOISE | BASELINE_SPARSE | 42 | 216.5 | 0.650 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| CLEAN | AC_FULL_SPARSE | 0 | AC_FULL_SPARSE_CLEAN_seed0_final_policy.pt | CLEAN | 288.5 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 3 | AC_FULL_SPARSE_CLEAN_seed3_final_policy.pt | CLEAN | 288.0 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 9 | AC_FULL_SPARSE_CLEAN_seed9_step0010000_policy.pt | CLEAN | 286.2 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 17 | AC_FULL_SPARSE_CLEAN_seed17_step0020000_policy.pt | CLEAN | 289.1 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 42 | AC_FULL_SPARSE_CLEAN_seed42_step0010000_policy.pt | CLEAN | 289.0 | 1.000 |
| CLEAN | AC_LITE_DENSE | 0 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | AC_LITE_DENSE | 3 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | AC_LITE_DENSE | 9 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | AC_LITE_DENSE | 17 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | AC_LITE_DENSE | 42 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 0 | AC_LITE_SPARSE_CLEAN_seed0_step0020000_policy.pt | CLEAN | 288.0 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 3 | AC_LITE_SPARSE_CLEAN_seed3_final_policy.pt | CLEAN | 288.1 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 9 | AC_LITE_SPARSE_CLEAN_seed9_final_policy.pt | CLEAN | 288.6 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 17 | AC_LITE_SPARSE_CLEAN_seed17_step0010000_policy.pt | CLEAN | 289.6 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 42 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | BASELINE_DENSE | 0 | BASELINE_DENSE_CLEAN_seed0_step0010000.pt | CLEAN | 287.9 | 1.000 |
| CLEAN | BASELINE_DENSE | 3 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | BASELINE_DENSE | 9 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | BASELINE_DENSE | 17 | BASELINE_DENSE_CLEAN_seed17_step0020000.pt | CLEAN | 288.4 | 1.000 |
| CLEAN | BASELINE_DENSE | 42 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | BASELINE_SPARSE | 0 | BASELINE_SPARSE_CLEAN_seed0_step0010000.pt | CLEAN | 287.6 | 1.000 |
| CLEAN | BASELINE_SPARSE | 3 | BASELINE_SPARSE_CLEAN_seed3_step0020000.pt | CLEAN | 288.2 | 1.000 |
| CLEAN | BASELINE_SPARSE | 9 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | BASELINE_SPARSE | 17 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | BASELINE_SPARSE | 42 | BASELINE_SPARSE_CLEAN_seed42_final.pt | CLEAN | 289.7 | 1.000 |
| OBS_NOISE | AC_FULL_SPARSE | 0 | AC_FULL_SPARSE_OBS_NOISE_seed0_final_policy.pt | OBS_NOISE | 233.7 | 0.733 |
| OBS_NOISE | AC_FULL_SPARSE | 3 | AC_FULL_SPARSE_OBS_NOISE_seed3_final_policy.pt | OBS_NOISE | 240.0 | 0.800 |
| OBS_NOISE | AC_FULL_SPARSE | 9 | checkpoint_0_pretrained | OBS_NOISE | 213.0 | 0.600 |
| OBS_NOISE | AC_FULL_SPARSE | 17 | AC_FULL_SPARSE_OBS_NOISE_seed17_final_policy.pt | OBS_NOISE | 265.3 | 0.867 |
| OBS_NOISE | AC_FULL_SPARSE | 42 | AC_FULL_SPARSE_OBS_NOISE_seed42_step0020000_policy.pt | OBS_NOISE | 218.2 | 0.733 |
| OBS_NOISE | AC_LITE_DENSE | 0 | AC_LITE_DENSE_OBS_NOISE_seed0_step0010000_policy.pt | OBS_NOISE | 247.3 | 0.800 |
| OBS_NOISE | AC_LITE_DENSE | 3 | AC_LITE_DENSE_OBS_NOISE_seed3_final_policy.pt | OBS_NOISE | 244.3 | 0.733 |
| OBS_NOISE | AC_LITE_DENSE | 9 | AC_LITE_DENSE_OBS_NOISE_seed9_final_policy.pt | OBS_NOISE | 239.1 | 0.733 |
| OBS_NOISE | AC_LITE_DENSE | 17 | AC_LITE_DENSE_OBS_NOISE_seed17_step0020000_policy.pt | OBS_NOISE | 234.5 | 0.667 |
| OBS_NOISE | AC_LITE_DENSE | 42 | AC_LITE_DENSE_OBS_NOISE_seed42_final_policy.pt | OBS_NOISE | 247.2 | 0.800 |
| OBS_NOISE | AC_LITE_SPARSE | 0 | AC_LITE_SPARSE_OBS_NOISE_seed0_final_policy.pt | OBS_NOISE | 252.6 | 0.800 |
| OBS_NOISE | AC_LITE_SPARSE | 3 | checkpoint_0_pretrained | OBS_NOISE | 213.0 | 0.600 |
| OBS_NOISE | AC_LITE_SPARSE | 9 | checkpoint_0_pretrained | OBS_NOISE | 213.0 | 0.600 |
| OBS_NOISE | AC_LITE_SPARSE | 17 | AC_LITE_SPARSE_OBS_NOISE_seed17_step0010000_policy.pt | OBS_NOISE | 238.3 | 0.667 |
| OBS_NOISE | AC_LITE_SPARSE | 42 | AC_LITE_SPARSE_OBS_NOISE_seed42_final_policy.pt | OBS_NOISE | 217.8 | 0.667 |
| OBS_NOISE | BASELINE_DENSE | 0 | BASELINE_DENSE_OBS_NOISE_seed0_step0020000.pt | OBS_NOISE | 250.9 | 0.800 |
| OBS_NOISE | BASELINE_DENSE | 3 | BASELINE_DENSE_OBS_NOISE_seed3_final.pt | OBS_NOISE | 258.3 | 0.867 |
| OBS_NOISE | BASELINE_DENSE | 9 | BASELINE_DENSE_OBS_NOISE_seed9_final.pt | OBS_NOISE | 222.1 | 0.600 |
| OBS_NOISE | BASELINE_DENSE | 17 | BASELINE_DENSE_OBS_NOISE_seed17_step0020000.pt | OBS_NOISE | 254.3 | 0.800 |
| OBS_NOISE | BASELINE_DENSE | 42 | BASELINE_DENSE_OBS_NOISE_seed42_final.pt | OBS_NOISE | 220.2 | 0.600 |
| OBS_NOISE | BASELINE_SPARSE | 0 | BASELINE_SPARSE_OBS_NOISE_seed0_final.pt | OBS_NOISE | 264.9 | 0.867 |
| OBS_NOISE | BASELINE_SPARSE | 3 | BASELINE_SPARSE_OBS_NOISE_seed3_final.pt | OBS_NOISE | 213.6 | 0.733 |
| OBS_NOISE | BASELINE_SPARSE | 9 | BASELINE_SPARSE_OBS_NOISE_seed9_final.pt | OBS_NOISE | 260.3 | 0.867 |
| OBS_NOISE | BASELINE_SPARSE | 17 | BASELINE_SPARSE_OBS_NOISE_seed17_final.pt | OBS_NOISE | 232.2 | 0.733 |
| OBS_NOISE | BASELINE_SPARSE | 42 | BASELINE_SPARSE_OBS_NOISE_seed42_final.pt | OBS_NOISE | 261.3 | 0.867 |
| REWARD_NOISE | AC_FULL_SPARSE | 0 | AC_FULL_SPARSE_REWARD_NOISE_seed0_step0010000_policy.pt | REWARD_NOISE | 287.5 | 1.000 |
| REWARD_NOISE | AC_FULL_SPARSE | 3 | AC_FULL_SPARSE_REWARD_NOISE_seed3_step0010000_policy.pt | REWARD_NOISE | 289.5 | 1.000 |
| REWARD_NOISE | AC_FULL_SPARSE | 9 | AC_FULL_SPARSE_REWARD_NOISE_seed9_step0020000_policy.pt | REWARD_NOISE | 289.2 | 1.000 |
| REWARD_NOISE | AC_FULL_SPARSE | 17 | AC_FULL_SPARSE_REWARD_NOISE_seed17_step0010000_policy.pt | REWARD_NOISE | 288.3 | 1.000 |
| REWARD_NOISE | AC_FULL_SPARSE | 42 | AC_FULL_SPARSE_REWARD_NOISE_seed42_final_policy.pt | REWARD_NOISE | 288.1 | 1.000 |
| REWARD_NOISE | AC_LITE_DENSE | 0 | AC_LITE_DENSE_REWARD_NOISE_seed0_step0010000_policy.pt | REWARD_NOISE | 287.7 | 1.000 |
| REWARD_NOISE | AC_LITE_DENSE | 3 | AC_LITE_DENSE_REWARD_NOISE_seed3_step0010000_policy.pt | REWARD_NOISE | 286.2 | 1.000 |
| REWARD_NOISE | AC_LITE_DENSE | 9 | AC_LITE_DENSE_REWARD_NOISE_seed9_step0010000_policy.pt | REWARD_NOISE | 286.6 | 1.000 |
| REWARD_NOISE | AC_LITE_DENSE | 17 | AC_LITE_DENSE_REWARD_NOISE_seed17_step0010000_policy.pt | REWARD_NOISE | 288.4 | 1.000 |
| REWARD_NOISE | AC_LITE_DENSE | 42 | AC_LITE_DENSE_REWARD_NOISE_seed42_step0020000_policy.pt | REWARD_NOISE | 287.8 | 1.000 |
| REWARD_NOISE | AC_LITE_SPARSE | 0 | AC_LITE_SPARSE_REWARD_NOISE_seed0_step0020000_policy.pt | REWARD_NOISE | 288.1 | 1.000 |
| REWARD_NOISE | AC_LITE_SPARSE | 3 | AC_LITE_SPARSE_REWARD_NOISE_seed3_step0010000_policy.pt | REWARD_NOISE | 289.2 | 1.000 |
| REWARD_NOISE | AC_LITE_SPARSE | 9 | AC_LITE_SPARSE_REWARD_NOISE_seed9_step0020000_policy.pt | REWARD_NOISE | 286.6 | 1.000 |
| REWARD_NOISE | AC_LITE_SPARSE | 17 | AC_LITE_SPARSE_REWARD_NOISE_seed17_step0010000_policy.pt | REWARD_NOISE | 289.8 | 1.000 |
| REWARD_NOISE | AC_LITE_SPARSE | 42 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE_DENSE | 0 | BASELINE_DENSE_REWARD_NOISE_seed0_step0010000.pt | REWARD_NOISE | 287.9 | 1.000 |
| REWARD_NOISE | BASELINE_DENSE | 3 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE_DENSE | 9 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE_DENSE | 17 | BASELINE_DENSE_REWARD_NOISE_seed17_final.pt | REWARD_NOISE | 288.0 | 1.000 |
| REWARD_NOISE | BASELINE_DENSE | 42 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE_SPARSE | 0 | BASELINE_SPARSE_REWARD_NOISE_seed0_step0010000.pt | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE_SPARSE | 3 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE_SPARSE | 9 | BASELINE_SPARSE_REWARD_NOISE_seed9_step0010000.pt | REWARD_NOISE | 290.9 | 1.000 |
| REWARD_NOISE | BASELINE_SPARSE | 17 | BASELINE_SPARSE_REWARD_NOISE_seed17_step0010000.pt | REWARD_NOISE | 289.3 | 1.000 |
| REWARD_NOISE | BASELINE_SPARSE | 42 | BASELINE_SPARSE_REWARD_NOISE_seed42_step0020000.pt | REWARD_NOISE | 289.3 | 1.000 |

## Best-checkpoint challenge tests

| training mode | method | test condition | eval mode | obs sigma | return | success |
|---|---|---|---|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | test_clean | CLEAN | 0.10 | 278.6 ± 19.2 | 1.000 ± 0.000 |
| CLEAN | AC_FULL_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 211.0 ± 108.3 | 0.713 ± 0.453 |
| CLEAN | AC_FULL_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 24.0 ± 35.1 | 0.000 ± 0.000 |
| CLEAN | AC_LITE_DENSE | test_clean | CLEAN | 0.10 | 277.4 ± 20.6 | 1.000 ± 0.000 |
| CLEAN | AC_LITE_DENSE | test_obs_noise | OBS_NOISE | 0.10 | 212.1 ± 110.7 | 0.710 ± 0.454 |
| CLEAN | AC_LITE_DENSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 21.7 ± 30.8 | 0.000 ± 0.000 |
| CLEAN | AC_LITE_SPARSE | test_clean | CLEAN | 0.10 | 278.5 ± 20.1 | 1.000 ± 0.000 |
| CLEAN | AC_LITE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 208.9 ± 109.8 | 0.693 ± 0.462 |
| CLEAN | AC_LITE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 21.0 ± 33.1 | 0.000 ± 0.000 |
| CLEAN | BASELINE_DENSE | test_clean | CLEAN | 0.10 | 277.9 ± 20.8 | 1.000 ± 0.000 |
| CLEAN | BASELINE_DENSE | test_obs_noise | OBS_NOISE | 0.10 | 220.2 ± 105.3 | 0.746 ± 0.436 |
| CLEAN | BASELINE_DENSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 24.8 ± 36.3 | 0.000 ± 0.000 |
| CLEAN | BASELINE_SPARSE | test_clean | CLEAN | 0.10 | 277.7 ± 19.6 | 1.000 ± 0.000 |
| CLEAN | BASELINE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 205.9 ± 112.9 | 0.693 ± 0.462 |
| CLEAN | BASELINE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 22.2 ± 31.3 | 0.000 ± 0.000 |
| OBS_NOISE | AC_FULL_SPARSE | test_clean | CLEAN | 0.10 | 266.9 ± 46.7 | 0.950 ± 0.218 |
| OBS_NOISE | AC_FULL_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 221.0 ± 106.0 | 0.745 ± 0.437 |
| OBS_NOISE | AC_FULL_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 34.8 ± 54.1 | 0.035 ± 0.184 |
| OBS_NOISE | AC_LITE_DENSE | test_clean | CLEAN | 0.10 | 276.4 ± 20.4 | 1.000 ± 0.000 |
| OBS_NOISE | AC_LITE_DENSE | test_obs_noise | OBS_NOISE | 0.10 | 250.4 ± 76.8 | 0.860 ± 0.348 |
| OBS_NOISE | AC_LITE_DENSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 39.3 ± 49.9 | 0.010 ± 0.100 |
| OBS_NOISE | AC_LITE_SPARSE | test_clean | CLEAN | 0.10 | 270.8 ± 41.8 | 0.967 ± 0.180 |
| OBS_NOISE | AC_LITE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 204.4 ± 115.0 | 0.677 ± 0.469 |
| OBS_NOISE | AC_LITE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 21.5 ± 33.8 | 0.003 ± 0.058 |
| OBS_NOISE | BASELINE_DENSE | test_clean | CLEAN | 0.10 | 273.7 ± 21.1 | 1.000 ± 0.000 |
| OBS_NOISE | BASELINE_DENSE | test_obs_noise | OBS_NOISE | 0.10 | 242.0 ± 86.9 | 0.830 ± 0.377 |
| OBS_NOISE | BASELINE_DENSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 40.3 ± 44.1 | 0.000 ± 0.000 |
| REWARD_NOISE | AC_FULL_SPARSE | test_clean | CLEAN | 0.10 | 275.0 ± 34.1 | 0.975 ± 0.156 |
| REWARD_NOISE | AC_FULL_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 172.7 ± 122.1 | 0.555 ± 0.498 |
| REWARD_NOISE | AC_FULL_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 17.1 ± 32.6 | 0.000 ± 0.000 |
| REWARD_NOISE | AC_LITE_DENSE | test_clean | CLEAN | 0.10 | 276.7 ± 21.3 | 1.000 ± 0.000 |
| REWARD_NOISE | AC_LITE_DENSE | test_obs_noise | OBS_NOISE | 0.10 | 230.9 ± 94.7 | 0.792 ± 0.406 |
| REWARD_NOISE | AC_LITE_DENSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 29.7 ± 42.1 | 0.006 ± 0.077 |
| REWARD_NOISE | AC_LITE_SPARSE | test_clean | CLEAN | 0.10 | 278.8 ± 19.4 | 1.000 ± 0.000 |
| REWARD_NOISE | AC_LITE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 166.6 ± 124.2 | 0.526 ± 0.500 |
| REWARD_NOISE | AC_LITE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 12.9 ± 29.1 | 0.000 ± 0.000 |
| REWARD_NOISE | BASELINE_DENSE | test_clean | CLEAN | 0.10 | 277.6 ± 20.8 | 1.000 ± 0.000 |
| REWARD_NOISE | BASELINE_DENSE | test_obs_noise | OBS_NOISE | 0.10 | 215.6 ± 108.3 | 0.725 ± 0.447 |
| REWARD_NOISE | BASELINE_DENSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 24.6 ± 35.0 | 0.000 ± 0.000 |
| REWARD_NOISE | BASELINE_SPARSE | test_clean | CLEAN | 0.10 | 274.7 ± 31.8 | 0.970 ± 0.171 |
| REWARD_NOISE | BASELINE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 195.2 ± 116.8 | 0.636 ± 0.482 |
| REWARD_NOISE | BASELINE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 21.4 ± 33.5 | 0.000 ± 0.000 |

## Episode-level certainty summary

| mode | method | mean c_i | corr(c, delta) | corr(c, action prob) | corr(c, runner-up prob) |
|---|---|---:|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | 0.534 | 0.097 | 0.112 | -0.100 |
| CLEAN | AC_LITE_DENSE | 0.531 | 0.137 | 0.156 | -0.132 |
| CLEAN | AC_LITE_SPARSE | 0.533 | 0.143 | 0.162 | -0.142 |
| OBS_NOISE | AC_FULL_SPARSE | 0.522 | 0.124 | 0.133 | -0.122 |
| OBS_NOISE | AC_LITE_DENSE | 0.531 | 0.050 | 0.055 | -0.048 |
| OBS_NOISE | AC_LITE_SPARSE | 0.536 | -0.014 | -0.015 | 0.014 |
| REWARD_NOISE | AC_FULL_SPARSE | 0.532 | 0.091 | 0.105 | -0.095 |
| REWARD_NOISE | AC_LITE_DENSE | 0.532 | 0.150 | 0.172 | -0.146 |
| REWARD_NOISE | AC_LITE_SPARSE | 0.534 | 0.065 | 0.078 | -0.067 |

## Certainty AUROC diagnostics

| mode | method | trajectory AUROC | timestep AUROC |
|---|---|---:|---:|
| CLEAN | AC_FULL_SPARSE | 0.485 | 0.459 |
| CLEAN | AC_LITE_DENSE | 0.436 | 0.445 |
| CLEAN | AC_LITE_SPARSE | 0.490 | 0.446 |
| OBS_NOISE | AC_FULL_SPARSE | 0.602 | 0.429 |
| OBS_NOISE | AC_LITE_DENSE | 0.580 | 0.480 |
| OBS_NOISE | AC_LITE_SPARSE | 0.405 | 0.511 |
| REWARD_NOISE | AC_FULL_SPARSE | 0.413 | 0.459 |
| REWARD_NOISE | AC_LITE_DENSE | 0.469 | 0.445 |
| REWARD_NOISE | AC_LITE_SPARSE | 0.461 | 0.468 |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- CLEAN: `AC_FULL_SPARSE` has the highest mean final return (268.6) with mean final success 0.940.
- OBS_NOISE: `AC_LITE_DENSE` has the highest mean final return (211.6) with mean final success 0.710.
- REWARD_NOISE: `BASELINE_DENSE` has the highest mean final return (266.1) with mean final success 0.720.

Checkpoint selection versus the pretrained anchor:

- CLEAN / AC_FULL_SPARSE: checkpoint 0 wins in 0 of 5 seeds (0.00).
- CLEAN / AC_LITE_DENSE: checkpoint 0 wins in 5 of 5 seeds (1.00).
- CLEAN / AC_LITE_SPARSE: checkpoint 0 wins in 1 of 5 seeds (0.20).
- CLEAN / BASELINE_DENSE: checkpoint 0 wins in 3 of 5 seeds (0.60).
- CLEAN / BASELINE_SPARSE: checkpoint 0 wins in 2 of 5 seeds (0.40).
- OBS_NOISE / AC_FULL_SPARSE: checkpoint 0 wins in 1 of 5 seeds (0.20).
- OBS_NOISE / AC_LITE_DENSE: checkpoint 0 wins in 0 of 5 seeds (0.00).
- OBS_NOISE / AC_LITE_SPARSE: checkpoint 0 wins in 2 of 5 seeds (0.40).
- OBS_NOISE / BASELINE_DENSE: checkpoint 0 wins in 0 of 5 seeds (0.00).
- OBS_NOISE / BASELINE_SPARSE: checkpoint 0 wins in 0 of 5 seeds (0.00).
- REWARD_NOISE / AC_FULL_SPARSE: checkpoint 0 wins in 0 of 5 seeds (0.00).
- REWARD_NOISE / AC_LITE_DENSE: checkpoint 0 wins in 0 of 5 seeds (0.00).
- REWARD_NOISE / AC_LITE_SPARSE: checkpoint 0 wins in 1 of 5 seeds (0.20).
- REWARD_NOISE / BASELINE_DENSE: checkpoint 0 wins in 3 of 5 seeds (0.60).
- REWARD_NOISE / BASELINE_SPARSE: checkpoint 0 wins in 1 of 5 seeds (0.20).

Best-checkpoint challenge testing:

- test_clean (CLEAN, sigma=0.10): best mean return is `AC_LITE_SPARSE` in `REWARD_NOISE` with 278.8 ± 19.4 and success 1.000 ± 0.000.
- test_obs_noise (OBS_NOISE, sigma=0.10): best mean return is `AC_LITE_DENSE` in `OBS_NOISE` with 250.4 ± 76.8 and success 0.860 ± 0.348.
- test_obs_noise_hard (OBS_NOISE, sigma=0.20): best mean return is `BASELINE_DENSE` in `OBS_NOISE` with 40.3 ± 44.1 and success 0.000 ± 0.000.

Episode-level certainty behavior:

- CLEAN / AC_FULL_SPARSE: mean episode certainty 0.534, mean corr(certainty, delta) 0.097, mean corr(certainty, action_prob) 0.112, mean corr(certainty, runner_up_prob) -0.100.
- CLEAN / AC_LITE_DENSE: mean episode certainty 0.531, mean corr(certainty, delta) 0.137, mean corr(certainty, action_prob) 0.156, mean corr(certainty, runner_up_prob) -0.132.
- CLEAN / AC_LITE_SPARSE: mean episode certainty 0.533, mean corr(certainty, delta) 0.143, mean corr(certainty, action_prob) 0.162, mean corr(certainty, runner_up_prob) -0.142.
- OBS_NOISE / AC_FULL_SPARSE: mean episode certainty 0.522, mean corr(certainty, delta) 0.124, mean corr(certainty, action_prob) 0.133, mean corr(certainty, runner_up_prob) -0.122.
- OBS_NOISE / AC_LITE_DENSE: mean episode certainty 0.531, mean corr(certainty, delta) 0.050, mean corr(certainty, action_prob) 0.055, mean corr(certainty, runner_up_prob) -0.048.
- OBS_NOISE / AC_LITE_SPARSE: mean episode certainty 0.536, mean corr(certainty, delta) -0.014, mean corr(certainty, action_prob) -0.015, mean corr(certainty, runner_up_prob) 0.014.
- REWARD_NOISE / AC_FULL_SPARSE: mean episode certainty 0.532, mean corr(certainty, delta) 0.091, mean corr(certainty, action_prob) 0.105, mean corr(certainty, runner_up_prob) -0.095.
- REWARD_NOISE / AC_LITE_DENSE: mean episode certainty 0.532, mean corr(certainty, delta) 0.150, mean corr(certainty, action_prob) 0.172, mean corr(certainty, runner_up_prob) -0.146.
- REWARD_NOISE / AC_LITE_SPARSE: mean episode certainty 0.534, mean corr(certainty, delta) 0.065, mean corr(certainty, action_prob) 0.078, mean corr(certainty, runner_up_prob) -0.067.

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

