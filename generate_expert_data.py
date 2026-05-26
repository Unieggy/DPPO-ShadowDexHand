"""
generate_expert_data.py

Phase A: Trains a SAC expert on HandManipulateEggDense-v1 using the same
          DiffusionStateNormalizer obs stack as the DPPO agent.
          Saves a resume checkpoint + replay buffer every CHECKPOINT_FREQ steps
          so Colab disconnects can be recovered without restarting from scratch.
Phase B: Records successful episodes into sliding-window action chunks and
          saves expert_data.pt: {"states": [N,75], "action_chunks": [N,4,20]}
"""

import sys
import os
import numpy as np
import torch
import gymnasium as gym
import gymnasium_robotics

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wrappers import DiffusionStateNormalizer

gym.register_envs(gymnasium_robotics)

# ── Config ─────────────────────────────────────────────────────────────────────
DENSE_ENV_ID       = "HandManipulateEggDense-v1"
CHUNK_SIZE         = 4
TARGET_TRANSITIONS = 100_000
SAC_TIMESTEPS      = 2_000_000
EVAL_FREQ          = 20_000       # evaluate success rate every N training steps
N_EVAL_EPISODES    = 20           # episodes per success-rate check
SUCCESS_THRESHOLD  = 0.95         # stop training once this is reached
SAC_SAVE_PATH      = "teacher_sac"
EXPERT_DATA_PATH   = "expert_data.pt"
RESUME_PATH        = "teacher_sac_resume"          # overwritten every CHECKPOINT_FREQ steps
RESUME_BUFFER_PATH = "teacher_sac_resume_buffer.pkl"
CHECKPOINT_FREQ    = 50_000       # save resume checkpoint every N steps
# ──────────────────────────────────────────────────────────────────────────────


def make_env(dense: bool = True):
    """
    Builds the env stack identical to DPPO's training env, minus chunking.
    SAC operates on single-step actions so ActionChunkingWrapper is NOT applied.
    RescaleAction ensures SAC's tanh outputs map cleanly to the env's torque range.
    """
    env_id = DENSE_ENV_ID if dense else "HandManipulateEgg-v1"
    env = gym.make(env_id)
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)   # [75]
    env = DiffusionStateNormalizer(env)              # online Welford normalisation → tanh
    env = gym.wrappers.RescaleAction(env, min_action=-1.0, max_action=1.0)
    return env


# ── Phase A: Train SAC expert ──────────────────────────────────────────────────

class StopOnSuccessRate(BaseCallback):
    """
    Evaluates success rate every eval_freq steps.
    Stops training and saves the best model when success_threshold is reached.
    """
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
                  f"{self.success_threshold*100:.1f}% target. Stopping SAC training.")
            return False

        return True


class ResumeCheckpointCallback(BaseCallback):
    """
    Saves model weights + replay buffer to a fixed path every checkpoint_freq steps.
    On Colab disconnect, re-running the script will load this and continue from here.
    """
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
                print(f"  [Resume checkpoint]  Saved @ {self.num_timesteps:,} steps "
                      f"→ {self.checkpoint_path}.zip + {self.buffer_path}")
        return True


def train_sac(resume: bool = False):
    print("=" * 60)
    print("PHASE A — Training SAC expert on dense-reward environment")
    if resume:
        print(f"  Resuming from {RESUME_PATH}.zip")
    print("=" * 60)

    train_env = Monitor(make_env(dense=True))
    eval_env  = make_env(dense=True)

    if resume:
        model = SAC.load(RESUME_PATH, env=train_env)
        if os.path.exists(RESUME_BUFFER_PATH):
            model.load_replay_buffer(RESUME_BUFFER_PATH)
            print(f"  Replay buffer loaded — {model.replay_buffer.size():,} transitions")
        steps_done    = model.num_timesteps
        steps_left    = SAC_TIMESTEPS - steps_done
        print(f"  Steps already completed: {steps_done:,} / {SAC_TIMESTEPS:,}")
        print(f"  Steps remaining        : {steps_left:,}")
        if steps_left <= 0:
            print("  Already reached total budget — skipping learn().")
            train_env.close()
            eval_env.close()
            best_model = SAC.load(SAC_SAVE_PATH, env=None)
            return best_model
    else:
        steps_left = SAC_TIMESTEPS
        model = SAC(
            "MlpPolicy",
            train_env,
            verbose          = 1,
            learning_rate    = 3e-4,
            batch_size       = 256,
            buffer_size      = 1_000_000,
            learning_starts  = 10_000,
            ent_coef         = "auto",
            gamma            = 0.98,
            tau              = 0.02,
            train_freq       = 1,
            gradient_steps   = 1,
            policy_kwargs    = dict(net_arch=[512, 512]),
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
        total_timesteps    = steps_left,
        callback           = callbacks,
        log_interval       = 20,
        reset_num_timesteps= not resume,
    )

    model.save(SAC_SAVE_PATH + "_final")
    print(f"\nFinal SAC model saved → {SAC_SAVE_PATH}_final.zip")
    train_env.close()
    eval_env.close()

    best_model = SAC.load(SAC_SAVE_PATH, env=None)
    print(f"Best model loaded from {SAC_SAVE_PATH}.zip")
    return best_model


# ── Phase B: Record expert demonstrations ─────────────────────────────────────

def record_expert_data(model):
    print("\n" + "=" * 60)
    print("PHASE B — Recording expert demonstrations")
    print(f"Target: {TARGET_TRANSITIONS:,} transitions from successful episodes")
    print("=" * 60)

    record_env = make_env(dense=True)

    all_states  = []
    all_chunks  = []

    total_transitions   = 0
    episode_count       = 0
    successful_episodes = 0

    while total_transitions < TARGET_TRANSITIONS:
        obs, _ = record_env.reset()
        done   = False
        info   = {}

        ep_states  = []
        ep_actions = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            ep_states.append(obs.copy())
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
            state = ep_states[i]
            chunk = np.stack(ep_actions[i : i + CHUNK_SIZE])
            all_states.append(state)
            all_chunks.append(chunk)
            total_transitions += 1
            new_this_episode  += 1

        print(f"  Episode {episode_count:>5}  ✓ success  "
              f"+{new_this_episode} transitions  "
              f"total: {total_transitions:,}/{TARGET_TRANSITIONS:,}")

        if episode_count % 50 == 0:
            yield_rate = successful_episodes / episode_count * 100
            print(f"\n  --- Progress summary ---")
            print(f"  Episodes run     : {episode_count}")
            print(f"  Success rate     : {yield_rate:.1f}%")
            print(f"  Transitions saved: {total_transitions:,}")
            print(f"  ------------------------\n")

    record_env.close()

    print(f"\nRecording complete.")
    print(f"  Total episodes     : {episode_count}")
    print(f"  Successful episodes: {successful_episodes}  "
          f"({successful_episodes/episode_count*100:.1f}%)")
    print(f"  Total transitions  : {total_transitions:,}")

    states_np = np.array(all_states, dtype=np.float32)
    chunks_np = np.array(all_chunks, dtype=np.float32)

    states_t = torch.from_numpy(states_np)
    chunks_t = torch.from_numpy(chunks_np)

    print(f"\nTensor shapes:")
    print(f"  states       : {states_t.shape}  dtype={states_t.dtype}")
    print(f"  action_chunks: {chunks_t.shape}  dtype={chunks_t.dtype}")

    torch.save({"states": states_t, "action_chunks": chunks_t}, EXPERT_DATA_PATH)
    print(f"\nSaved → {EXPERT_DATA_PATH}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if os.path.exists(SAC_SAVE_PATH + ".zip"):
        # Phase A fully done — best model already saved
        print(f"Found {SAC_SAVE_PATH}.zip — skipping Phase A.")
        model = SAC.load(SAC_SAVE_PATH)
    elif os.path.exists(RESUME_PATH + ".zip"):
        # Phase A was interrupted — resume from last checkpoint
        print(f"Found {RESUME_PATH}.zip — resuming Phase A from last checkpoint.")
        model = train_sac(resume=True)
    else:
        # Fresh start
        model = train_sac(resume=False)

    record_expert_data(model)
