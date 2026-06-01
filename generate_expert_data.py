"""
generate_expert_data.py

Phase A: Trains a SAC+HER expert on HandManipulateEgg-v1 (sparse reward).
          HER relabels failed episodes with achieved goals, flooding the replay
          buffer with synthetic successes so SAC gets a learning signal even when
          the policy almost never reaches the true goal.
          Saves a resume checkpoint + replay buffer every CHECKPOINT_FREQ steps.
Phase B: Records successful episodes into sliding-window action chunks.
          States are batch-normalized (mean/std + tanh) over all recorded data
          so the actor receives inputs in the same [-1,1] range as DPPO training.
          Saves expert_data.pt: {"states": [N,75], "action_chunks": [N,4,20]}
"""

import os
import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import HerReplayBuffer

gym.register_envs(gymnasium_robotics)

# ── Config ─────────────────────────────────────────────────────────────────────
ENV_ID             = "HandManipulateEgg-v1"   # sparse — HER provides the signal
CHUNK_SIZE         = 4
TARGET_TRANSITIONS = 100_000
SAC_TIMESTEPS      = 2_000_000
EVAL_FREQ          = 20_000
N_EVAL_EPISODES    = 20
SUCCESS_THRESHOLD  = 0.95
SAC_SAVE_PATH      = "teacher_sac"
EXPERT_DATA_PATH   = "expert_data.pt"
RESUME_PATH        = "teacher_sac_resume"
RESUME_BUFFER_PATH = "teacher_sac_resume_buffer.pkl"
CHECKPOINT_FREQ    = 50_000
# ──────────────────────────────────────────────────────────────────────────────


def make_sac_env():
    """
    Dict observation env for SAC+HER.
    HER requires the raw Dict structure {observation, achieved_goal, desired_goal}
    so it can relabel goals and recompute rewards on replayed transitions.
    MultiInputPolicy handles the Dict obs natively — no FlattenObservation needed.
    """
    env = gym.make(ENV_ID)
    env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)
    return env


# ── Phase A callbacks ──────────────────────────────────────────────────────────

class StopOnSuccessRate(BaseCallback):
    """Evaluates every eval_freq steps; saves best model; stops at threshold."""
    def __init__(self, eval_env, eval_freq, success_threshold,
                 n_eval_episodes=20, save_path=SAC_SAVE_PATH, verbose=1):
        super().__init__(verbose)
        self.eval_env          = eval_env
        self.eval_freq         = eval_freq
        self.success_threshold = success_threshold
        self.n_eval_episodes   = n_eval_episodes
        self.save_path         = save_path
        self.best_rate         = 0.0

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        successes = 0
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            info = {}
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = self.eval_env.step(action)
                done = terminated or truncated
            if info.get("is_success", False):
                successes += 1

        rate = successes / self.n_eval_episodes
        if self.verbose:
            print(f"\n  [Eval @ {self.num_timesteps:,} steps]  "
                  f"Success rate: {rate*100:.1f}%  "
                  f"(best so far: {self.best_rate*100:.1f}%)")

        if rate > self.best_rate:
            self.best_rate = rate
            self.model.save(self.save_path)
            print(f"  [Saved]  New best model → {self.save_path}.zip")

        if rate >= self.success_threshold:
            print(f"\n  [Done]  Reached {rate*100:.1f}% ≥ "
                  f"{self.success_threshold*100:.1f}% target. Stopping.")
            return False

        return True


class ResumeCheckpointCallback(BaseCallback):
    """Overwrites a fixed checkpoint + replay buffer every checkpoint_freq steps."""
    def __init__(self, checkpoint_path, buffer_path, checkpoint_freq, verbose=1):
        super().__init__(verbose)
        self.checkpoint_path = checkpoint_path
        self.buffer_path     = buffer_path
        self.checkpoint_freq = checkpoint_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.checkpoint_freq == 0:
            self.model.save(self.checkpoint_path)
            self.model.save_replay_buffer(self.buffer_path)
            if self.verbose:
                print(f"  [Resume checkpoint]  @ {self.num_timesteps:,} steps "
                      f"→ {self.checkpoint_path}.zip")
        return True


# ── Phase A: Train SAC+HER expert ─────────────────────────────────────────────

def train_sac(resume: bool = False):
    print("=" * 60)
    print("PHASE A — SAC+HER on HandManipulateEgg-v1 (sparse reward)")
    if resume:
        print(f"  Resuming from {RESUME_PATH}.zip")
    print("=" * 60)

    train_env = Monitor(make_sac_env())
    eval_env  = make_sac_env()

    if resume:
        model = SAC.load(RESUME_PATH, env=train_env)
        if os.path.exists(RESUME_BUFFER_PATH):
            model.load_replay_buffer(RESUME_BUFFER_PATH)
            print(f"  Replay buffer loaded — {model.replay_buffer.size():,} transitions")
        steps_done = model.num_timesteps
        steps_left = SAC_TIMESTEPS - steps_done
        print(f"  Completed: {steps_done:,} / {SAC_TIMESTEPS:,}  —  {steps_left:,} remaining")
        if steps_left <= 0:
            print("  Budget exhausted — loading best saved model.")
            train_env.close()
            eval_env.close()
            return SAC.load(SAC_SAVE_PATH, env=None)
    else:
        steps_left = SAC_TIMESTEPS
        model = SAC(
            "MultiInputPolicy",           # handles Dict obs {obs, achieved_goal, desired_goal}
            train_env,
            replay_buffer_class  = HerReplayBuffer,
            replay_buffer_kwargs = dict(
                n_sampled_goal          = 4,       # 4 synthetic goals per real transition
                goal_selection_strategy = "future", # sample goals from later in same episode
            ),
            verbose         = 1,
            learning_rate   = 3e-4,
            batch_size      = 256,
            buffer_size     = 1_000_000,
            learning_starts = 1_000,
            ent_coef        = "auto",
            gamma           = 0.98,
            tau             = 0.02,
            train_freq      = 1,
            gradient_steps  = 1,
            policy_kwargs   = dict(net_arch=[512, 512]),
        )

    callbacks = CallbackList([
        StopOnSuccessRate(
            eval_env          = eval_env,
            eval_freq         = EVAL_FREQ,
            success_threshold = SUCCESS_THRESHOLD,
            n_eval_episodes   = N_EVAL_EPISODES,
            save_path         = SAC_SAVE_PATH,
        ),
        ResumeCheckpointCallback(
            checkpoint_path = RESUME_PATH,
            buffer_path     = RESUME_BUFFER_PATH,
            checkpoint_freq = CHECKPOINT_FREQ,
        ),
    ])

    model.learn(
        total_timesteps     = steps_left,
        callback            = callbacks,
        log_interval        = 20,
        reset_num_timesteps = not resume,
    )

    model.save(SAC_SAVE_PATH + "_final")
    print(f"\nFinal model saved → {SAC_SAVE_PATH}_final.zip")
    train_env.close()
    eval_env.close()

    return SAC.load(SAC_SAVE_PATH, env=None)


# ── Phase B: Record expert demonstrations ─────────────────────────────────────

def record_expert_data(model):
    print("\n" + "=" * 60)
    print("PHASE B — Recording expert demonstrations")
    print(f"Target: {TARGET_TRANSITIONS:,} transitions from successful episodes")
    print("=" * 60)

    record_env = make_sac_env()   # same Dict obs env the model was trained on

    all_states_raw = []   # raw flat [75] — batch-normalized at the end
    all_chunks     = []   # [4, 20] action chunks

    total_transitions   = 0
    episode_count       = 0
    successful_episodes = 0

    while total_transitions < TARGET_TRANSITIONS:
        obs, _ = record_env.reset()
        done   = False
        info   = {}

        ep_states_raw = []
        ep_actions    = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # Flatten Dict obs → raw [75] for storage
            flat = np.concatenate([
                obs["observation"],
                obs["achieved_goal"],
                obs["desired_goal"],
            ])
            ep_states_raw.append(flat)
            ep_actions.append(action.copy())
            obs, _, terminated, truncated, info = record_env.step(action)
            done = terminated or truncated

        episode_count += 1

        if not info.get("is_success", False):
            continue

        successful_episodes += 1
        n = len(ep_actions)

        new_this_episode = 0
        for i in range(n - CHUNK_SIZE + 1):
            all_states_raw.append(ep_states_raw[i])
            all_chunks.append(np.stack(ep_actions[i : i + CHUNK_SIZE]))
            total_transitions += 1
            new_this_episode  += 1

        print(f"  Episode {episode_count:>5}  ✓ success  "
              f"+{new_this_episode} transitions  "
              f"total: {total_transitions:,}/{TARGET_TRANSITIONS:,}")

        if episode_count % 50 == 0:
            print(f"\n  --- Progress ---")
            print(f"  Episodes run : {episode_count}")
            print(f"  Success rate : {successful_episodes/episode_count*100:.1f}%")
            print(f"  Transitions  : {total_transitions:,}")
            print(f"  ---------------\n")

    record_env.close()

    print(f"\nRecording complete.")
    print(f"  Successful episodes: {successful_episodes} / {episode_count} "
          f"({successful_episodes/episode_count*100:.1f}%)")

    # Batch-normalize: compute mean/std over all N transitions, then tanh
    # This puts states in [-1,1], matching DiffusionStateNormalizer's output range
    states_raw  = np.array(all_states_raw, dtype=np.float32)   # [N, 75]
    mean        = states_raw.mean(axis=0)
    std         = states_raw.std(axis=0) + 1e-8
    states_norm = np.tanh((states_raw - mean) / std).astype(np.float32)

    chunks_np = np.array(all_chunks, dtype=np.float32)         # [N, 4, 20]

    states_t = torch.from_numpy(states_norm)
    chunks_t = torch.from_numpy(chunks_np)

    print(f"\nTensor shapes:")
    print(f"  states       : {states_t.shape}  dtype={states_t.dtype}")
    print(f"  action_chunks: {chunks_t.shape}  dtype={chunks_t.dtype}")

    torch.save({"states": states_t, "action_chunks": chunks_t}, EXPERT_DATA_PATH)
    print(f"\nSaved → {EXPERT_DATA_PATH}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if os.path.exists(SAC_SAVE_PATH + ".zip"):
        print(f"Found {SAC_SAVE_PATH}.zip — skipping Phase A.")
        model = SAC.load(SAC_SAVE_PATH)
    elif os.path.exists(RESUME_PATH + ".zip"):
        print(f"Found {RESUME_PATH}.zip — resuming Phase A.")
        model = train_sac(resume=True)
    else:
        model = train_sac(resume=False)

    record_expert_data(model)
