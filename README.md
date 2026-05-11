# DPPO for Dexterous Manipulation

## Overview
This repository contains a custom implementation of Diffusion Policy Policy Optimization (DPPO) tailored for high-dimensional, state-based dexterous manipulation tasks. The primary target environment is the `gymnasium-robotics` Shadow Dexterous Hand. This implementation bridges generative diffusion models with reinforcement learning by fine-tuning a pre-trained diffusion policy to maximize task rewards using PPO.

## Architecture and Design
* Environment: `gymnasium-robotics` (Shadow Dexterous Hand)
* Observation Space: 1D State Vector (requires strict [-1, 1] normalization)
* Action Space: Continuous, High-dimensional (20+ Degrees of Freedom)
* Policy Format: Action Chunking (predicts Tp future steps, executes Ta steps)
* Actor Network: MLP Backbone with Sinusoidal Positional Embeddings for the diffusion time-step (k)
* Critic Network: Standard MLP predicting state value V(s)
* Diffusion Strategy: DDIM sampling, with RL fine-tuning applied exclusively to the last K' steps of the denoising process

## Project Structure
The implementation is modularized into four primary phases:

### 1. Environment Wrappers & Data Pipeline
* Chunking Wrapper: A custom Gym wrapper that accepts an action chunk of shape `[Ta, action_dim]`, executes actions sequentially in the base environment, aggregates rewards, and returns the final observation.
* Normalization Wrapper: Scales all incoming state observations and outgoing actions to a strictly bounded range to ensure numerical stability for the diffusion model.
* Reward Scaler: Stabilizes RL advantage calculations to prevent gradient explosion.

### 2. Neural Network Backbones
* Actor (Diffusion MLP): Processes `[state, noisy_action_chunk, k_step]`. Utilizes sinusoidal positional embeddings for the `k_step`, concatenates features, and outputs predicted noise matching the action chunk shape.
* Critic: A standard multi-layer perceptron outputting the scalar value estimate V(s).

### 3. DPPO Math & Buffer
* Dashcam Buffer: A custom rollout buffer designed to store intermediate states, noisy actions, refined actions, and log-probabilities for every k-step during the active fine-tuning window.
* Advantage Estimation (GAE): Computes the advantage at the end of the physical environment macro-step (t) and back-assigns it to the internal denoising micro-steps (k).
* PPO Objective: Implements the clipped surrogate loss function, calculating Gaussian log-likelihoods carefully to preserve numerical stability.

### 4. Training Loops
* Pre-training (Behavior Cloning): Supervised learning loop to train the initial MLP diffusion policy on demonstration data.
* RL Fine-Tuning (DPPO): The main reinforcement learning loop. The frozen model handles early denoising steps, while the active model handles the final K' steps, records transitions to the Dashcam Buffer, steps the environment, and executes the PPO optimization phase.

## References
* Original DPPO implementation and mathematical framework: https://github.com/irom-princeton/dppo