# DPPO for Dexterous Manipulation

## Overview
This repository contains a custom implementation of Diffusion Policy Policy Optimization (DPPO) tailored for high-dimensional, state-based dexterous manipulation tasks. The target environment is the `gymnasium-robotics` Shadow Dexterous Hand tasked with **in-hand egg reorientation** (`HandManipulateEgg-v1`). The egg was chosen over the block because its rotational symmetry around one axis reduces the number of hard-to-reach goal orientations, making it a more tractable first target for sparse-reward RL. This implementation bridges generative diffusion models with reinforcement learning by fine-tuning a pre-trained diffusion policy to maximize task rewards using PPO.

## Architecture and Design
* Environment: `HandManipulateEgg-v1` (Shadow Dexterous Hand, gymnasium-robotics) — sparse reward
* Expert Teacher: SAC + HER (`stable-baselines3`) trained on the same sparse env
* Observation Space: 1D State Vector (requires strict [-1, 1] normalization)
* Action Space: Continuous, High-dimensional (20 Degrees of Freedom)
* Policy Format: Action Chunking (predicts Tp future steps, executes Ta steps)
* Actor Network: MLP Backbone with Sinusoidal Positional Embeddings for the diffusion time-step (k)
* Critic Network: Standard MLP predicting state value V(s)
* Diffusion Strategy: DDIM sampling, with RL fine-tuning applied exclusively to the last K' steps of the denoising process

## Project Structure
The implementation is modularized into five primary components:

### 0. Expert Data Generation (`generate_expert_data.py`)
Must be run once before any DPPO training. Operates in two phases:
* **Phase A — SAC+HER Teacher**: Trains a SAC agent with **Hindsight Experience Replay** on `HandManipulateEgg-v1` (sparse reward) using `stable-baselines3`. HER relabels failed episodes with achieved goals, providing dense synthetic learning signal even when the true success rate is near zero. Uses `MultiInputPolicy` on the raw Dict observation space `{observation, achieved_goal, desired_goal}`. Evaluates every 20k steps, stops early at 95% success (up to 2M timesteps). Saves the best model as `teacher_sac.zip`. A resume checkpoint (`teacher_sac_resume.zip` + replay buffer) is overwritten every 50k steps — re-running the script automatically resumes from here if interrupted.
* **Phase B — Data Recording**: Runs the trained SAC deterministically on the same Dict obs env. Flattens each obs, collects only successful episodes, and formats actions into sliding-window chunks of size `Tp=4`. States are batch-normalized (mean/std + tanh) over all 100k recorded transitions before saving, putting them in `[-1, 1]` to match the DPPO actor's expected input range. Saves `expert_data.pt` with keys `{"states": [N,75], "action_chunks": [N,4,20]}`.
* Re-running skips Phase A automatically if `teacher_sac.zip` already exists.

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
* **`visualize.py`**: Loads `actor_best.pt`, runs deterministic evaluation episodes with `render_mode="rgb_array"`, and saves `eval_result.mp4`.

## Training Order
```
1. python generate_expert_data.py   # ~2-3 hours on Colab, run once
2. main.ipynb                        # BC pretraining → DPPO fine-tuning
3. python visualize.py               # render best policy to video
```

## Output Files
| File | When saved | Contents |
|---|---|---|
| `teacher_sac.zip` | Phase A best checkpoint | SAC+HER policy weights |
| `teacher_sac_resume.zip` | Every 50k SAC steps | Resume checkpoint (overwritten each time) |
| `teacher_sac_resume_buffer.pkl` | Every 50k SAC steps | HER replay buffer for resume |
| `expert_data.pt` | End of Phase B | `{states:[N,75], action_chunks:[N,4,20]}` |
| `actor_bc.pt` | After BC pretraining | Actor weights before RL |
| `actor_iter50.pt` ... | Every 50 DPPO iterations | Periodic safety checkpoints |
| `actor_best.pt` | When eval success rate improves | Best policy by success rate |
| `actor_final.pt` | End of 1000 iterations | Final policy weights |

## References
* Original DPPO implementation and mathematical framework: https://github.com/irom-princeton/dppo