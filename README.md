# DPPO for Dexterous Manipulation

## Overview
This repository contains a custom implementation of Diffusion Policy Policy Optimization (DPPO) tailored for high-dimensional, state-based dexterous manipulation tasks. The target environment is the `gymnasium-robotics` **Adroit Hand Pen task** (`AdroitHandPen-v1`), where a 24-DoF Shadow Hand must reorient a pen to match a randomized target orientation. Expert demonstrations are downloaded directly from the **Minari** offline RL dataset library (human demonstrations), bypassing the need to train an RL teacher. This implementation bridges generative diffusion models with reinforcement learning by fine-tuning a pre-trained diffusion policy to maximize task rewards using PPO.

## Architecture and Design
* Environment: `AdroitHandPen-v1` (Adroit Hand, gymnasium-robotics) — sparse reward (10.0 success / -0.1 otherwise)
* Expert Data: Human demonstrations downloaded from Minari (`pen-human-v0` or equivalent)
* Observation Space: 1D flat state vector, ~45 dims (verify with `env.observation_space.shape[0]`)
* Action Space: Continuous, 24 Degrees of Freedom (absolute joint angles, scaled to [-1, 1])
* Policy Format: Action Chunking (predicts Tp future steps, executes Ta steps)
* Actor Network: MLP Backbone with Sinusoidal Positional Embeddings for the diffusion time-step (k)
* Critic Network: Standard MLP predicting state value V(s)
* Diffusion Strategy: DDIM sampling, with RL fine-tuning applied exclusively to the last K' steps of the denoising process

## Project Structure
The implementation is modularized into five primary components:

### 0. Expert Data (`download_expert_data.py`)
Downloads and converts human demonstration data from Minari into the format expected by BC pretraining. Run once before any training.
* **Download**: Fetches `pen-human-v0` (or the available pen dataset) via `minari.load_dataset()`.
* **Conversion**: Extracts `(observation, action)` pairs from all episodes, creates sliding-window action chunks of size `Tp=4`, and batch-normalizes states (mean/std + tanh) to put them in `[-1, 1]`.
* **Output**: Saves `expert_data.pt` with keys `{"states": [N, obs_dim], "action_chunks": [N, 4, 24]}`.

### 1. Environment Wrappers (`wrappers.py`)
* **Chunking Wrapper**: Accepts an action chunk of shape `[Ta, action_dim]`, executes actions sequentially, aggregates rewards, and returns the final observation.
* **Normalization Wrapper**: Online Welford running mean/variance normalization followed by tanh squashing. Ensures all observations stay in `[-1, 1]` for diffusion model stability.
* **Reward Scaler**: Scales rewards by 0.01 to stabilize PPO advantage calculations.

### 2. Neural Network Backbones (`network.py`)
* **Actor (Diffusion MLP)**: Processes `[state, noisy_action_chunk, k_step]`. Sinusoidal positional embeddings encode the diffusion timestep `k`, which is concatenated with the flattened state and noisy action before passing through a 4-layer Mish MLP. Outputs predicted noise of shape `[chunk_size, act_dim]`.
* **Critic**: Standard MLP mapping state → scalar value `V(s)`. No k-step conditioning.

### 3. DPPO Math & Buffer (`dppo_math.py`)
* **Dashcam Buffer**: Preallocated buffer storing `T×K'` micro-step transitions. Stores `x_k` (noisy actions), `committed_noises` (sampled ε), k-steps, log-probs, advantages, and returns.
* **Advantage Estimation**: Single-step `A_t = r_t − V(s_t)` broadcast to all K' denoising steps of the same environment step.
* **PPO Objective**: Clipped surrogate loss with Gaussian log-likelihoods computed in noise-prediction space.
* **Log Variance Helper**: Extracts `log(β_k)` from the DDPM scheduler's noise schedule for policy variance.

### 4. Training Loops (`train.py`)
* **Behavior Cloning**: Supervised pretraining on `expert_data.pt`. Corrupts expert actions at random diffusion steps via `scheduler.add_noise()` and trains the actor to predict the added noise via MSE.
* **Evaluation**: Deterministic rollout (no exploration noise) every 20 iterations. Tracks `is_success` from the environment. Saves `actor_best.pt` when success rate improves.
* **DPPO Fine-Tuning**: K→0 denoising rollout with stochastic committed-noise sampling, PPO update over 4 epochs, periodic checkpointing every 50 iterations directly to Google Drive.

### 5. Entry Points
* **`main.ipynb`**: Colab notebook. Mounts Google Drive, copies project files, installs dependencies. Checks for `expert_data.pt` and runs BC pretraining if found, then runs DPPO. All checkpoints saved to `MyDrive/dppo/checkpoints/`.
* **`visualize.py`**: Loads checkpoints, runs deterministic evaluation, saves `eval_result.mp4`. Also generates the **BC vs DPPO comparison plot** — success rate of the BC-only policy vs the DPPO fine-tuned policy to demonstrate RL improvement.

## Demonstrating DPPO Effectiveness
The standard comparison used in the DPPO paper is:

| Policy | Description |
|---|---|
| **BC only** | `actor_bc.pt` — pure imitation, no RL |
| **BC + DPPO** | `actor_best.pt` — BC init fine-tuned with PPO |

`visualize.py` evaluates both checkpoints over 20 episodes each and plots success rate side-by-side. A meaningful result shows DPPO pushing success rate above the BC baseline, demonstrating that RL fine-tuning on the diffusion denoising steps extracts additional task performance beyond what imitation alone achieves.

## Training Order
```
1. python download_expert_data.py   # fetch Minari data, convert to expert_data.pt
2. main.ipynb                        # BC pretraining → DPPO fine-tuning
3. python visualize.py               # render video + BC vs DPPO comparison plot
```

## Output Files
| File | When saved | Contents |
|---|---|---|
| `expert_data.pt` | After download script | `{states:[N,obs_dim], action_chunks:[N,4,24]}` |
| `actor_bc.pt` | After BC pretraining | Actor weights before RL (BC baseline) |
| `actor_iter50.pt` ... | Every 50 DPPO iterations | Periodic safety checkpoints |
| `actor_best.pt` | When eval success rate improves | Best policy by success rate |
| `actor_final.pt` | End of 1000 iterations | Final policy weights |

## References
* Original DPPO implementation and mathematical framework: https://github.com/irom-princeton/dppo
* Adroit Hand demonstrations: Rajeswaran et al., "Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations" (2018)
