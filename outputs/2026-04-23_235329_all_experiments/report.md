# RL Experiment Report

This report summarizes the selected sweep from the generated CSV logs.

Source folder: `G:\ac-grpo\outputs\2026-04-23_235329_all_experiments`

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
- Challenge tests currently use up to 2540 episodes per evaluation seed in the generated logs.

## Summary table (mean ± std over seeds)

| mode | method | final return (last 20 eps) | final success (last 20 eps) | reward AUC | best rolling-20 return | best rolling-20 success |
|---|---|---:|---:|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | 273.4 ± 14.9 | 0.940 ± 0.082 | 266.1 ± 5.8 | 293.7 | 1.000 |
| CLEAN | AC_LITE_DENSE | 254.6 ± 26.6 | 0.890 ± 0.089 | 244.5 ± 17.7 | 286.7 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 275.8 ± 5.5 | 0.990 ± 0.022 | 257.2 ± 8.8 | 294.0 | 1.000 |
| CLEAN | BASELINE_DENSE | 243.5 ± 22.3 | 0.830 ± 0.115 | 230.9 ± 7.2 | 289.4 | 1.000 |
| CLEAN | BASELINE_SPARSE | 272.1 ± 11.3 | 0.930 ± 0.045 | 249.6 ± 4.1 | 294.2 | 1.000 |
| OBS_NOISE | AC_FULL_SPARSE | 151.9 ± 136.4 | 0.650 ± 0.355 | 205.0 ± 28.7 | 284.8 | 1.000 |
| OBS_NOISE | AC_LITE_DENSE | 182.1 ± 29.9 | 0.530 ± 0.315 | 230.9 ± 13.3 | 285.9 | 1.000 |
| OBS_NOISE | AC_LITE_SPARSE | 209.9 ± 28.4 | 0.760 ± 0.114 | 221.8 ± 5.3 | 285.8 | 1.000 |
| OBS_NOISE | BASELINE_DENSE | 51.2 ± 34.4 | 0.060 ± 0.134 | 148.4 ± 20.1 | 282.7 | 1.000 |
| OBS_NOISE | BASELINE_SPARSE | 206.2 ± 40.0 | 0.770 ± 0.125 | 225.1 ± 9.9 | 287.6 | 1.000 |
| REWARD_NOISE | AC_FULL_SPARSE | 225.9 ± 43.3 | 0.540 ± 0.134 | 231.9 ± 12.7 | 290.2 | 0.950 |
| REWARD_NOISE | AC_LITE_DENSE | 242.0 ± 31.3 | 0.690 ± 0.139 | 233.7 ± 18.9 | 289.8 | 0.970 |
| REWARD_NOISE | AC_LITE_SPARSE | 269.5 ± 9.2 | 0.840 ± 0.114 | 236.0 ± 20.5 | 292.6 | 0.980 |
| REWARD_NOISE | BASELINE_DENSE | 235.0 ± 47.3 | 0.630 ± 0.164 | 232.6 ± 11.8 | 289.2 | 0.970 |
| REWARD_NOISE | BASELINE_SPARSE | 211.6 ± 86.7 | 0.570 ± 0.327 | 228.5 ± 29.5 | 291.1 | 0.970 |

## Per-seed finals (last 20 episodes)

| mode | method | seed | final return | final success |
|---|---|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | 0 | 280.7 | 0.950 |
| CLEAN | AC_FULL_SPARSE | 3 | 285.9 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 9 | 247.8 | 0.800 |
| CLEAN | AC_FULL_SPARSE | 17 | 274.9 | 0.950 |
| CLEAN | AC_FULL_SPARSE | 42 | 277.9 | 1.000 |
| CLEAN | AC_LITE_DENSE | 0 | 248.4 | 0.850 |
| CLEAN | AC_LITE_DENSE | 3 | 259.3 | 0.950 |
| CLEAN | AC_LITE_DENSE | 9 | 212.6 | 0.750 |
| CLEAN | AC_LITE_DENSE | 17 | 281.6 | 0.950 |
| CLEAN | AC_LITE_DENSE | 42 | 270.9 | 0.950 |
| CLEAN | AC_LITE_SPARSE | 0 | 271.6 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 3 | 279.2 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 9 | 269.7 | 0.950 |
| CLEAN | AC_LITE_SPARSE | 17 | 283.1 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 42 | 275.6 | 1.000 |
| CLEAN | BASELINE_DENSE | 0 | 259.3 | 0.900 |
| CLEAN | BASELINE_DENSE | 3 | 232.0 | 0.750 |
| CLEAN | BASELINE_DENSE | 9 | 227.3 | 0.750 |
| CLEAN | BASELINE_DENSE | 17 | 224.3 | 0.750 |
| CLEAN | BASELINE_DENSE | 42 | 274.7 | 1.000 |
| CLEAN | BASELINE_SPARSE | 0 | 262.1 | 0.900 |
| CLEAN | BASELINE_SPARSE | 3 | 282.0 | 0.950 |
| CLEAN | BASELINE_SPARSE | 9 | 261.9 | 0.900 |
| CLEAN | BASELINE_SPARSE | 17 | 286.2 | 1.000 |
| CLEAN | BASELINE_SPARSE | 42 | 268.3 | 0.900 |
| OBS_NOISE | AC_FULL_SPARSE | 0 | 234.3 | 0.850 |
| OBS_NOISE | AC_FULL_SPARSE | 3 | -84.6 | 0.050 |
| OBS_NOISE | AC_FULL_SPARSE | 9 | 154.4 | 0.600 |
| OBS_NOISE | AC_FULL_SPARSE | 17 | 218.9 | 0.850 |
| OBS_NOISE | AC_FULL_SPARSE | 42 | 236.6 | 0.900 |
| OBS_NOISE | AC_LITE_DENSE | 0 | 219.8 | 0.900 |
| OBS_NOISE | AC_LITE_DENSE | 3 | 161.0 | 0.500 |
| OBS_NOISE | AC_LITE_DENSE | 9 | 173.6 | 0.500 |
| OBS_NOISE | AC_LITE_DENSE | 17 | 149.6 | 0.050 |
| OBS_NOISE | AC_LITE_DENSE | 42 | 206.5 | 0.700 |
| OBS_NOISE | AC_LITE_SPARSE | 0 | 240.1 | 0.900 |
| OBS_NOISE | AC_LITE_SPARSE | 3 | 174.3 | 0.650 |
| OBS_NOISE | AC_LITE_SPARSE | 9 | 195.3 | 0.750 |
| OBS_NOISE | AC_LITE_SPARSE | 17 | 237.8 | 0.850 |
| OBS_NOISE | AC_LITE_SPARSE | 42 | 202.0 | 0.650 |
| OBS_NOISE | BASELINE_DENSE | 0 | 64.4 | 0.000 |
| OBS_NOISE | BASELINE_DENSE | 3 | 84.1 | 0.300 |
| OBS_NOISE | BASELINE_DENSE | 9 | 22.8 | 0.000 |
| OBS_NOISE | BASELINE_DENSE | 17 | 78.0 | 0.000 |
| OBS_NOISE | BASELINE_DENSE | 42 | 7.0 | 0.000 |
| OBS_NOISE | BASELINE_SPARSE | 0 | 254.9 | 0.900 |
| OBS_NOISE | BASELINE_SPARSE | 3 | 167.3 | 0.650 |
| OBS_NOISE | BASELINE_SPARSE | 9 | 184.4 | 0.750 |
| OBS_NOISE | BASELINE_SPARSE | 17 | 243.6 | 0.900 |
| OBS_NOISE | BASELINE_SPARSE | 42 | 180.6 | 0.650 |
| REWARD_NOISE | AC_FULL_SPARSE | 0 | 194.3 | 0.450 |
| REWARD_NOISE | AC_FULL_SPARSE | 3 | 275.1 | 0.600 |
| REWARD_NOISE | AC_FULL_SPARSE | 9 | 228.7 | 0.750 |
| REWARD_NOISE | AC_FULL_SPARSE | 17 | 171.7 | 0.450 |
| REWARD_NOISE | AC_FULL_SPARSE | 42 | 259.7 | 0.450 |
| REWARD_NOISE | AC_LITE_DENSE | 0 | 218.2 | 0.550 |
| REWARD_NOISE | AC_LITE_DENSE | 3 | 261.1 | 0.700 |
| REWARD_NOISE | AC_LITE_DENSE | 9 | 274.1 | 0.800 |
| REWARD_NOISE | AC_LITE_DENSE | 17 | 256.7 | 0.850 |
| REWARD_NOISE | AC_LITE_DENSE | 42 | 200.2 | 0.550 |
| REWARD_NOISE | AC_LITE_SPARSE | 0 | 269.7 | 0.850 |
| REWARD_NOISE | AC_LITE_SPARSE | 3 | 270.5 | 0.950 |
| REWARD_NOISE | AC_LITE_SPARSE | 9 | 258.8 | 0.700 |
| REWARD_NOISE | AC_LITE_SPARSE | 17 | 283.6 | 0.950 |
| REWARD_NOISE | AC_LITE_SPARSE | 42 | 264.8 | 0.750 |
| REWARD_NOISE | BASELINE_DENSE | 0 | 244.2 | 0.600 |
| REWARD_NOISE | BASELINE_DENSE | 3 | 277.5 | 0.850 |
| REWARD_NOISE | BASELINE_DENSE | 9 | 244.2 | 0.600 |
| REWARD_NOISE | BASELINE_DENSE | 17 | 153.9 | 0.400 |
| REWARD_NOISE | BASELINE_DENSE | 42 | 255.2 | 0.700 |
| REWARD_NOISE | BASELINE_SPARSE | 0 | 252.9 | 0.700 |
| REWARD_NOISE | BASELINE_SPARSE | 3 | 249.4 | 0.600 |
| REWARD_NOISE | BASELINE_SPARSE | 9 | 243.2 | 0.750 |
| REWARD_NOISE | BASELINE_SPARSE | 17 | 56.9 | 0.000 |
| REWARD_NOISE | BASELINE_SPARSE | 42 | 255.9 | 0.800 |

## Best checkpoint by primary greedy held-out selection

| mode | method | seed | checkpoint | eval mode | eval return | eval success |
|---|---|---:|---|---|---:|---:|
| CLEAN | AC_FULL_SPARSE | 0 | AC_FULL_SPARSE_CLEAN_seed0_step0400000_policy.pt | CLEAN | 287.7 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 3 | AC_FULL_SPARSE_CLEAN_seed3_step0600000_policy.pt | CLEAN | 289.0 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 9 | AC_FULL_SPARSE_CLEAN_seed9_step0650000_policy.pt | CLEAN | 288.0 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 17 | AC_FULL_SPARSE_CLEAN_seed17_step0650000_policy.pt | CLEAN | 289.9 | 1.000 |
| CLEAN | AC_FULL_SPARSE | 42 | AC_FULL_SPARSE_CLEAN_seed42_step0300000_policy.pt | CLEAN | 286.9 | 1.000 |
| CLEAN | AC_LITE_DENSE | 0 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | AC_LITE_DENSE | 3 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | AC_LITE_DENSE | 9 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | AC_LITE_DENSE | 17 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | AC_LITE_DENSE | 42 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 0 | AC_LITE_SPARSE_CLEAN_seed0_final_policy.pt | CLEAN | 287.3 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 3 | AC_LITE_SPARSE_CLEAN_seed3_step0100000_policy.pt | CLEAN | 286.9 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 9 | AC_LITE_SPARSE_CLEAN_seed9_step0950000_policy.pt | CLEAN | 287.9 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 17 | AC_LITE_SPARSE_CLEAN_seed17_step0950000_policy.pt | CLEAN | 287.4 | 1.000 |
| CLEAN | AC_LITE_SPARSE | 42 | AC_LITE_SPARSE_CLEAN_seed42_step0050000_policy.pt | CLEAN | 289.2 | 1.000 |
| CLEAN | BASELINE_DENSE | 0 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | BASELINE_DENSE | 3 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | BASELINE_DENSE | 9 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | BASELINE_DENSE | 17 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | BASELINE_DENSE | 42 | checkpoint_0_pretrained | CLEAN | 285.9 | 1.000 |
| CLEAN | BASELINE_SPARSE | 0 | BASELINE_SPARSE_CLEAN_seed0_step0900000.pt | CLEAN | 290.7 | 1.000 |
| CLEAN | BASELINE_SPARSE | 3 | BASELINE_SPARSE_CLEAN_seed3_step0750000.pt | CLEAN | 288.7 | 1.000 |
| CLEAN | BASELINE_SPARSE | 9 | BASELINE_SPARSE_CLEAN_seed9_step0850000.pt | CLEAN | 289.7 | 1.000 |
| CLEAN | BASELINE_SPARSE | 17 | BASELINE_SPARSE_CLEAN_seed17_step0900000.pt | CLEAN | 292.0 | 1.000 |
| CLEAN | BASELINE_SPARSE | 42 | BASELINE_SPARSE_CLEAN_seed42_step0950000.pt | CLEAN | 289.4 | 1.000 |
| OBS_NOISE | AC_FULL_SPARSE | 0 | AC_FULL_SPARSE_OBS_NOISE_seed0_step0350000_policy.pt | OBS_NOISE | 285.5 | 1.000 |
| OBS_NOISE | AC_FULL_SPARSE | 3 | AC_FULL_SPARSE_OBS_NOISE_seed3_step0500000_policy.pt | OBS_NOISE | 279.0 | 1.000 |
| OBS_NOISE | AC_FULL_SPARSE | 9 | AC_FULL_SPARSE_OBS_NOISE_seed9_step0050000_policy.pt | OBS_NOISE | 285.0 | 1.000 |
| OBS_NOISE | AC_FULL_SPARSE | 17 | AC_FULL_SPARSE_OBS_NOISE_seed17_step0200000_policy.pt | OBS_NOISE | 284.8 | 1.000 |
| OBS_NOISE | AC_FULL_SPARSE | 42 | AC_FULL_SPARSE_OBS_NOISE_seed42_step0750000_policy.pt | OBS_NOISE | 285.3 | 1.000 |
| OBS_NOISE | AC_LITE_DENSE | 0 | AC_LITE_DENSE_OBS_NOISE_seed0_step0450000_policy.pt | OBS_NOISE | 273.1 | 1.000 |
| OBS_NOISE | AC_LITE_DENSE | 3 | AC_LITE_DENSE_OBS_NOISE_seed3_step0200000_policy.pt | OBS_NOISE | 277.2 | 1.000 |
| OBS_NOISE | AC_LITE_DENSE | 9 | AC_LITE_DENSE_OBS_NOISE_seed9_step0150000_policy.pt | OBS_NOISE | 286.8 | 1.000 |
| OBS_NOISE | AC_LITE_DENSE | 17 | AC_LITE_DENSE_OBS_NOISE_seed17_step0200000_policy.pt | OBS_NOISE | 277.7 | 1.000 |
| OBS_NOISE | AC_LITE_DENSE | 42 | AC_LITE_DENSE_OBS_NOISE_seed42_step0100000_policy.pt | OBS_NOISE | 280.1 | 1.000 |
| OBS_NOISE | AC_LITE_SPARSE | 0 | AC_LITE_SPARSE_OBS_NOISE_seed0_step0400000_policy.pt | OBS_NOISE | 287.7 | 1.000 |
| OBS_NOISE | AC_LITE_SPARSE | 3 | AC_LITE_SPARSE_OBS_NOISE_seed3_step0750000_policy.pt | OBS_NOISE | 281.4 | 1.000 |
| OBS_NOISE | AC_LITE_SPARSE | 9 | AC_LITE_SPARSE_OBS_NOISE_seed9_step0100000_policy.pt | OBS_NOISE | 285.7 | 1.000 |
| OBS_NOISE | AC_LITE_SPARSE | 17 | AC_LITE_SPARSE_OBS_NOISE_seed17_step0700000_policy.pt | OBS_NOISE | 282.5 | 1.000 |
| OBS_NOISE | AC_LITE_SPARSE | 42 | AC_LITE_SPARSE_OBS_NOISE_seed42_step0150000_policy.pt | OBS_NOISE | 276.9 | 1.000 |
| OBS_NOISE | BASELINE_DENSE | 0 | BASELINE_DENSE_OBS_NOISE_seed0_step0100000.pt | OBS_NOISE | 277.7 | 0.933 |
| OBS_NOISE | BASELINE_DENSE | 3 | BASELINE_DENSE_OBS_NOISE_seed3_step0250000.pt | OBS_NOISE | 269.6 | 0.933 |
| OBS_NOISE | BASELINE_DENSE | 9 | BASELINE_DENSE_OBS_NOISE_seed9_step0200000.pt | OBS_NOISE | 275.2 | 1.000 |
| OBS_NOISE | BASELINE_DENSE | 17 | BASELINE_DENSE_OBS_NOISE_seed17_step0200000.pt | OBS_NOISE | 274.5 | 1.000 |
| OBS_NOISE | BASELINE_DENSE | 42 | BASELINE_DENSE_OBS_NOISE_seed42_step0100000.pt | OBS_NOISE | 277.8 | 1.000 |
| OBS_NOISE | BASELINE_SPARSE | 0 | BASELINE_SPARSE_OBS_NOISE_seed0_step0450000.pt | OBS_NOISE | 283.9 | 1.000 |
| OBS_NOISE | BASELINE_SPARSE | 3 | BASELINE_SPARSE_OBS_NOISE_seed3_step0150000.pt | OBS_NOISE | 286.2 | 1.000 |
| OBS_NOISE | BASELINE_SPARSE | 9 | BASELINE_SPARSE_OBS_NOISE_seed9_step0700000.pt | OBS_NOISE | 279.4 | 1.000 |
| OBS_NOISE | BASELINE_SPARSE | 17 | BASELINE_SPARSE_OBS_NOISE_seed17_step0200000.pt | OBS_NOISE | 286.0 | 1.000 |
| OBS_NOISE | BASELINE_SPARSE | 42 | BASELINE_SPARSE_OBS_NOISE_seed42_step0300000.pt | OBS_NOISE | 283.7 | 1.000 |
| REWARD_NOISE | AC_FULL_SPARSE | 0 | AC_FULL_SPARSE_REWARD_NOISE_seed0_final_policy.pt | REWARD_NOISE | 291.8 | 1.000 |
| REWARD_NOISE | AC_FULL_SPARSE | 3 | AC_FULL_SPARSE_REWARD_NOISE_seed3_step0850000_policy.pt | REWARD_NOISE | 286.6 | 1.000 |
| REWARD_NOISE | AC_FULL_SPARSE | 9 | AC_FULL_SPARSE_REWARD_NOISE_seed9_step0750000_policy.pt | REWARD_NOISE | 286.2 | 1.000 |
| REWARD_NOISE | AC_FULL_SPARSE | 17 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | AC_FULL_SPARSE | 42 | AC_FULL_SPARSE_REWARD_NOISE_seed42_step0600000_policy.pt | REWARD_NOISE | 293.3 | 1.000 |
| REWARD_NOISE | AC_LITE_DENSE | 0 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | AC_LITE_DENSE | 3 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | AC_LITE_DENSE | 9 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | AC_LITE_DENSE | 17 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | AC_LITE_DENSE | 42 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | AC_LITE_SPARSE | 0 | AC_LITE_SPARSE_REWARD_NOISE_seed0_step0650000_policy.pt | REWARD_NOISE | 290.3 | 1.000 |
| REWARD_NOISE | AC_LITE_SPARSE | 3 | AC_LITE_SPARSE_REWARD_NOISE_seed3_step0750000_policy.pt | REWARD_NOISE | 288.7 | 1.000 |
| REWARD_NOISE | AC_LITE_SPARSE | 9 | AC_LITE_SPARSE_REWARD_NOISE_seed9_step0950000_policy.pt | REWARD_NOISE | 291.4 | 1.000 |
| REWARD_NOISE | AC_LITE_SPARSE | 17 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | AC_LITE_SPARSE | 42 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE_DENSE | 0 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE_DENSE | 3 | BASELINE_DENSE_REWARD_NOISE_seed3_step0050000.pt | REWARD_NOISE | 288.0 | 1.000 |
| REWARD_NOISE | BASELINE_DENSE | 9 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE_DENSE | 17 | BASELINE_DENSE_REWARD_NOISE_seed17_step0050000.pt | REWARD_NOISE | 287.2 | 1.000 |
| REWARD_NOISE | BASELINE_DENSE | 42 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE_SPARSE | 0 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE_SPARSE | 3 | BASELINE_SPARSE_REWARD_NOISE_seed3_final.pt | REWARD_NOISE | 290.7 | 1.000 |
| REWARD_NOISE | BASELINE_SPARSE | 9 | checkpoint_0_pretrained | REWARD_NOISE | 285.9 | 1.000 |
| REWARD_NOISE | BASELINE_SPARSE | 17 | BASELINE_SPARSE_REWARD_NOISE_seed17_step0100000.pt | REWARD_NOISE | 287.0 | 1.000 |
| REWARD_NOISE | BASELINE_SPARSE | 42 | BASELINE_SPARSE_REWARD_NOISE_seed42_step0050000.pt | REWARD_NOISE | 286.2 | 1.000 |

## Best-checkpoint challenge tests

| training mode | method | test condition | eval mode | obs sigma | return | success |
|---|---|---|---|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | test_clean | CLEAN | 0.10 | 275.0 ± 29.2 | 0.980 ± 0.140 |
| CLEAN | AC_FULL_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 156.8 ± 81.7 | 0.308 ± 0.462 |
| CLEAN | AC_FULL_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 36.0 ± 49.5 | 0.000 ± 0.000 |
| CLEAN | AC_LITE_DENSE | test_clean | CLEAN | 0.10 | 277.4 ± 20.6 | 1.000 ± 0.000 |
| CLEAN | AC_LITE_DENSE | test_obs_noise | OBS_NOISE | 0.10 | 212.1 ± 110.7 | 0.710 ± 0.454 |
| CLEAN | AC_LITE_DENSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 21.7 ± 30.8 | 0.000 ± 0.000 |
| CLEAN | AC_LITE_SPARSE | test_clean | CLEAN | 0.10 | 275.4 ± 22.2 | 0.988 ± 0.111 |
| CLEAN | AC_LITE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 161.4 ± 101.2 | 0.388 ± 0.488 |
| CLEAN | AC_LITE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 25.0 ± 35.8 | 0.000 ± 0.000 |
| CLEAN | BASELINE_DENSE | test_clean | CLEAN | 0.10 | 277.4 ± 20.6 | 1.000 ± 0.000 |
| CLEAN | BASELINE_DENSE | test_obs_noise | OBS_NOISE | 0.10 | 212.1 ± 110.7 | 0.710 ± 0.454 |
| CLEAN | BASELINE_DENSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 21.7 ± 30.8 | 0.000 ± 0.000 |
| CLEAN | BASELINE_SPARSE | test_clean | CLEAN | 0.10 | 274.0 ± 30.5 | 0.968 ± 0.176 |
| CLEAN | BASELINE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 79.7 ± 82.1 | 0.096 ± 0.295 |
| CLEAN | BASELINE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | -2.1 ± 32.5 | 0.000 ± 0.000 |
| OBS_NOISE | AC_FULL_SPARSE | test_clean | CLEAN | 0.10 | 241.4 ± 71.9 | 0.846 ± 0.361 |
| OBS_NOISE | AC_FULL_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 251.5 ± 85.2 | 0.896 ± 0.306 |
| OBS_NOISE | AC_FULL_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 160.0 ± 134.7 | 0.562 ± 0.497 |
| OBS_NOISE | AC_LITE_DENSE | test_clean | CLEAN | 0.10 | 202.8 ± 98.3 | 0.758 ± 0.429 |
| OBS_NOISE | AC_LITE_DENSE | test_obs_noise | OBS_NOISE | 0.10 | 264.8 ± 39.2 | 0.960 ± 0.196 |
| OBS_NOISE | AC_LITE_DENSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 149.9 ± 95.2 | 0.336 ± 0.473 |
| OBS_NOISE | AC_LITE_SPARSE | test_clean | CLEAN | 0.10 | 258.2 ± 35.6 | 0.926 ± 0.262 |
| OBS_NOISE | AC_LITE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 262.3 ± 70.6 | 0.950 ± 0.218 |
| OBS_NOISE | AC_LITE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 154.3 ± 131.4 | 0.534 ± 0.499 |
| OBS_NOISE | BASELINE_DENSE | test_clean | CLEAN | 0.10 | 253.4 ± 26.4 | 0.970 ± 0.171 |
| OBS_NOISE | BASELINE_DENSE | test_obs_noise | OBS_NOISE | 0.10 | 264.6 ± 42.3 | 0.962 ± 0.191 |
| OBS_NOISE | BASELINE_DENSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 175.6 ± 112.2 | 0.558 ± 0.497 |
| OBS_NOISE | BASELINE_SPARSE | test_clean | CLEAN | 0.10 | 260.4 ± 47.4 | 0.946 ± 0.226 |
| OBS_NOISE | BASELINE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 260.5 ± 72.3 | 0.936 ± 0.245 |
| OBS_NOISE | BASELINE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 154.7 ± 136.4 | 0.538 ± 0.499 |
| REWARD_NOISE | AC_FULL_SPARSE | test_clean | CLEAN | 0.10 | 276.7 ± 25.6 | 0.988 ± 0.111 |
| REWARD_NOISE | AC_FULL_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 95.2 ± 132.4 | 0.210 ± 0.408 |
| REWARD_NOISE | AC_FULL_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | -6.2 ± 72.0 | 0.000 ± 0.000 |
| REWARD_NOISE | AC_LITE_DENSE | test_clean | CLEAN | 0.10 | 277.4 ± 20.6 | 1.000 ± 0.000 |
| REWARD_NOISE | AC_LITE_DENSE | test_obs_noise | OBS_NOISE | 0.10 | 212.1 ± 110.7 | 0.710 ± 0.454 |
| REWARD_NOISE | AC_LITE_DENSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 21.7 ± 30.8 | 0.000 ± 0.000 |
| REWARD_NOISE | AC_LITE_SPARSE | test_clean | CLEAN | 0.10 | 279.7 ± 20.1 | 1.000 ± 0.000 |
| REWARD_NOISE | AC_LITE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 120.8 ± 109.2 | 0.284 ± 0.451 |
| REWARD_NOISE | AC_LITE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 3.6 ± 34.8 | 0.000 ± 0.000 |
| REWARD_NOISE | BASELINE_DENSE | test_clean | CLEAN | 0.10 | 277.2 ± 20.5 | 1.000 ± 0.000 |
| REWARD_NOISE | BASELINE_DENSE | test_obs_noise | OBS_NOISE | 0.10 | 214.4 ± 108.7 | 0.718 ± 0.450 |
| REWARD_NOISE | BASELINE_DENSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 24.7 ± 33.3 | 0.000 ± 0.000 |
| REWARD_NOISE | BASELINE_SPARSE | test_clean | CLEAN | 0.10 | 275.0 ± 24.5 | 0.988 ± 0.111 |
| REWARD_NOISE | BASELINE_SPARSE | test_obs_noise | OBS_NOISE | 0.10 | 193.5 ± 111.5 | 0.632 ± 0.483 |
| REWARD_NOISE | BASELINE_SPARSE | test_obs_noise_hard | OBS_NOISE | 0.20 | 18.7 ± 30.8 | 0.000 ± 0.000 |

## Cross-test summary on OBS evaluation

| Method | Train: CLEAN → Test OBS | Train: OBS → Test OBS | Train: REWARD → Test OBS |
|---|---:|---:|---:|
| Baseline (Sparse) | 79.7 ± 82.1 / 0.096 ± 0.295 | 260.5 ± 72.3 / 0.936 ± 0.245 | 193.5 ± 111.5 / 0.632 ± 0.483 |
| Baseline (Dense) | 212.1 ± 110.7 / 0.710 ± 0.454 | 264.6 ± 42.3 / 0.962 ± 0.191 | 214.4 ± 108.7 / 0.718 ± 0.450 |
| AC-LITE (Sparse) | 161.4 ± 101.2 / 0.388 ± 0.488 | 262.3 ± 70.6 / 0.950 ± 0.218 | 120.8 ± 109.2 / 0.284 ± 0.451 |
| AC-LITE (Dense) | 212.1 ± 110.7 / 0.710 ± 0.454 | 264.8 ± 39.2 / 0.960 ± 0.196 | 212.1 ± 110.7 / 0.710 ± 0.454 |
| AC-FULL (Sparse) | 156.8 ± 81.7 / 0.308 ± 0.462 | 251.5 ± 85.2 / 0.896 ± 0.306 | 95.2 ± 132.4 / 0.210 ± 0.408 |

## Episode-level certainty summary

| mode | method | mean c_i | corr(c, delta) | corr(c, action prob) | corr(c, runner-up prob) |
|---|---|---:|---:|---:|---:|
| CLEAN | AC_FULL_SPARSE | 0.867 | -0.079 | -0.071 | 0.092 |
| CLEAN | AC_LITE_DENSE | 0.859 | 0.333 | 0.388 | -0.310 |
| CLEAN | AC_LITE_SPARSE | 0.898 | 0.315 | 0.355 | -0.301 |
| OBS_NOISE | AC_FULL_SPARSE | 0.716 | 0.407 | 0.448 | -0.375 |
| OBS_NOISE | AC_LITE_DENSE | 0.904 | 0.208 | 0.233 | -0.196 |
| OBS_NOISE | AC_LITE_SPARSE | 0.910 | 0.298 | 0.324 | -0.290 |
| REWARD_NOISE | AC_FULL_SPARSE | 0.678 | 0.408 | 0.454 | -0.381 |
| REWARD_NOISE | AC_LITE_DENSE | 0.866 | 0.315 | 0.379 | -0.287 |
| REWARD_NOISE | AC_LITE_SPARSE | 0.886 | 0.324 | 0.362 | -0.305 |

## Result analysis

The final training-window summary shows the following strongest method per mode:

- CLEAN: `AC_LITE_SPARSE` has the highest mean final return (275.8) with mean final success 0.990.
- OBS_NOISE: `AC_LITE_SPARSE` has the highest mean final return (209.9) with mean final success 0.760.
- REWARD_NOISE: `AC_LITE_SPARSE` has the highest mean final return (269.5) with mean final success 0.840.

Checkpoint selection versus the pretrained anchor:

- CLEAN / AC_FULL_SPARSE: checkpoint 0 wins in 0 of 5 seeds (0.00).
- CLEAN / AC_LITE_DENSE: checkpoint 0 wins in 5 of 5 seeds (1.00).
- CLEAN / AC_LITE_SPARSE: checkpoint 0 wins in 0 of 5 seeds (0.00).
- CLEAN / BASELINE_DENSE: checkpoint 0 wins in 5 of 5 seeds (1.00).
- CLEAN / BASELINE_SPARSE: checkpoint 0 wins in 0 of 5 seeds (0.00).
- OBS_NOISE / AC_FULL_SPARSE: checkpoint 0 wins in 0 of 5 seeds (0.00).
- OBS_NOISE / AC_LITE_DENSE: checkpoint 0 wins in 0 of 5 seeds (0.00).
- OBS_NOISE / AC_LITE_SPARSE: checkpoint 0 wins in 0 of 5 seeds (0.00).
- OBS_NOISE / BASELINE_DENSE: checkpoint 0 wins in 0 of 5 seeds (0.00).
- OBS_NOISE / BASELINE_SPARSE: checkpoint 0 wins in 0 of 5 seeds (0.00).
- REWARD_NOISE / AC_FULL_SPARSE: checkpoint 0 wins in 1 of 5 seeds (0.20).
- REWARD_NOISE / AC_LITE_DENSE: checkpoint 0 wins in 5 of 5 seeds (1.00).
- REWARD_NOISE / AC_LITE_SPARSE: checkpoint 0 wins in 2 of 5 seeds (0.40).
- REWARD_NOISE / BASELINE_DENSE: checkpoint 0 wins in 3 of 5 seeds (0.60).
- REWARD_NOISE / BASELINE_SPARSE: checkpoint 0 wins in 2 of 5 seeds (0.40).

Best-checkpoint challenge testing:

- test_clean (CLEAN, sigma=0.10): best mean return is `AC_LITE_SPARSE` in `REWARD_NOISE` with 279.7 ± 20.1 and success 1.000 ± 0.000.
- test_obs_noise (OBS_NOISE, sigma=0.10): best mean return is `AC_LITE_DENSE` in `OBS_NOISE` with 264.8 ± 39.2 and success 0.960 ± 0.196.
- test_obs_noise_hard (OBS_NOISE, sigma=0.20): best mean return is `BASELINE_DENSE` in `OBS_NOISE` with 175.6 ± 112.2 and success 0.558 ± 0.497.

Episode-level certainty behavior:

- CLEAN / AC_FULL_SPARSE: mean episode certainty 0.867, mean corr(certainty, delta) -0.079, mean corr(certainty, action_prob) -0.071, mean corr(certainty, runner_up_prob) 0.092.
- CLEAN / AC_LITE_DENSE: mean episode certainty 0.859, mean corr(certainty, delta) 0.333, mean corr(certainty, action_prob) 0.388, mean corr(certainty, runner_up_prob) -0.310.
- CLEAN / AC_LITE_SPARSE: mean episode certainty 0.898, mean corr(certainty, delta) 0.315, mean corr(certainty, action_prob) 0.355, mean corr(certainty, runner_up_prob) -0.301.
- OBS_NOISE / AC_FULL_SPARSE: mean episode certainty 0.716, mean corr(certainty, delta) 0.407, mean corr(certainty, action_prob) 0.448, mean corr(certainty, runner_up_prob) -0.375.
- OBS_NOISE / AC_LITE_DENSE: mean episode certainty 0.904, mean corr(certainty, delta) 0.208, mean corr(certainty, action_prob) 0.233, mean corr(certainty, runner_up_prob) -0.196.
- OBS_NOISE / AC_LITE_SPARSE: mean episode certainty 0.910, mean corr(certainty, delta) 0.298, mean corr(certainty, action_prob) 0.324, mean corr(certainty, runner_up_prob) -0.290.
- REWARD_NOISE / AC_FULL_SPARSE: mean episode certainty 0.678, mean corr(certainty, delta) 0.408, mean corr(certainty, action_prob) 0.454, mean corr(certainty, runner_up_prob) -0.381.
- REWARD_NOISE / AC_LITE_DENSE: mean episode certainty 0.866, mean corr(certainty, delta) 0.315, mean corr(certainty, action_prob) 0.379, mean corr(certainty, runner_up_prob) -0.287.
- REWARD_NOISE / AC_LITE_SPARSE: mean episode certainty 0.886, mean corr(certainty, delta) 0.324, mean corr(certainty, action_prob) 0.362, mean corr(certainty, runner_up_prob) -0.305.

## Plots

The following plots are generated in the `plots/` subfolder:

1. `01_return_vs_steps.png`
2. `02_success_rate_vs_steps.png`
3. `03_certainty_histogram.png`
4. `04_certainty_vs_entropy_scatter.png`
5. `05_certainty_vs_delta_t_scatter.png`
6. `06_return_by_mode_subplots.png`
7. `07_success_by_mode_subplots.png`

