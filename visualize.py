import torch
import numpy as np
import imageio
import gymnasium as gym
import gymnasium_robotics  # noqa: F401 — registers HandManipulate envs
from diffusers import DDPMScheduler

from network import DiffusionMLPActor
from wrappers import DiffusionStateNormalizer

# ── Config ────────────────────────────────────────────────────────────────────
ENV_ID       = "AdriotHandPen-v1"
OBS_DIM      = 75
ACT_DIM      = 20
CHUNK_SIZE   = 4
K            = 100
HIDDEN_DIM   = 256
CHECKPOINT   = "checkpoints/actor_final.pt"
OUTPUT_VIDEO = "eval_result.mp4"
NUM_EPISODES = 3
FPS          =30
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────


def build_eval_env():
    # Build env manually so we can pass render_mode.
    # Apply the same obs wrappers as training but NOT ActionChunkingWrapper
    # since we step through the chunk ourselves to capture every frame.
    env = gym.make(ENV_ID, render_mode="rgb_array")
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)
    env = DiffusionStateNormalizer(env)
    return env


def run_denoising_chain(actor, scheduler, obs_tensor):
    """Full K→0 reverse diffusion pass. Returns clean action chunk [chunk_size, act_dim]."""
    x = torch.randn(1, actor.chunk_size, actor.act_dim, device=DEVICE)
    scheduler.set_timesteps(scheduler.config.num_train_timesteps)

    with torch.no_grad():
        for k in scheduler.timesteps:          # iterates K-1 → 0
            k_tensor   = torch.tensor([k], dtype=torch.long, device=DEVICE)
            noise_pred = actor(obs_tensor, x, k_tensor)
            x = scheduler.step(noise_pred, k, x).prev_sample

    return x.squeeze(0).cpu().numpy()  # [chunk_size, act_dim]


def evaluate():
    print(f"Loading checkpoint: {CHECKPOINT}")
    actor = DiffusionMLPActor(OBS_DIM, ACT_DIM, CHUNK_SIZE, HIDDEN_DIM).to(DEVICE)
    actor.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    actor.eval()

    scheduler = DDPMScheduler(
        num_train_timesteps=K,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    env = build_eval_env()
    frames = []
    rewards = []

    for ep in range(NUM_EPISODES):
        obs, _       = env.reset()
        ep_reward    = 0.0
        done         = False

        while not done:
            obs_tensor   = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action_chunk = run_denoising_chain(actor, scheduler, obs_tensor)

            # Step through each action in the chunk, recording every frame
            for i in range(CHUNK_SIZE):
                obs, reward, terminated, truncated, _ = env.step(action_chunk[i])
                ep_reward += reward
                frames.append(env.render())
                if terminated or truncated:
                    done = True
                    break

        rewards.append(ep_reward)
        print(f"Episode {ep + 1}/{NUM_EPISODES}  reward = {ep_reward:.3f}")

    print(f"\nMean reward over {NUM_EPISODES} episodes: {np.mean(rewards):.3f}")

    if frames:
        imageio.mimsave(OUTPUT_VIDEO, frames, fps=FPS)
        print(f"Video saved → {OUTPUT_VIDEO}")

    env.close()


if __name__ == "__main__":
    evaluate()
