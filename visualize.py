import torch
import imageio
import gymnasium as gym
import gymnasium_robotics  # noqa: F401 — registers AdroitHand envs
from diffusers import DDPMScheduler

from network import DiffusionMLPActor
from wrappers import DiffusionStateNormalizer

# ── Config ────────────────────────────────────────────────────────────────────
ENV_ID       = "AdroitHandPen-v1"
OBS_DIM      = 45
ACT_DIM      = 24
CHUNK_SIZE   = 4
K            = 100
HIDDEN_DIM   = 256
BC_CHECKPOINT   = "checkpoints/actor_bc.pt"
BEST_CHECKPOINT = "checkpoints/actor_best.pt"
OUTPUT_VIDEO = "eval_result.mp4"
NUM_EPISODES = 20
FPS          = 30
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────


def build_eval_env():
    import os
    obs_mean = obs_std = None
    if os.path.exists("expert_data.pt"):
        _stats = torch.load("expert_data.pt", map_location="cpu")
        if "obs_mean" in _stats:
            obs_mean = _stats["obs_mean"].numpy()
            obs_std  = _stats["obs_std"].numpy()
    env = gym.make(ENV_ID, render_mode="rgb_array")
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = gym.wrappers.FlattenObservation(env)
    env = DiffusionStateNormalizer(env, mean=obs_mean, std=obs_std)
    return env


def load_actor(checkpoint_path):
    actor = DiffusionMLPActor(OBS_DIM, ACT_DIM, CHUNK_SIZE, HIDDEN_DIM).to(DEVICE)
    actor.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    actor.eval()
    return actor


def run_denoising_chain(actor, scheduler, obs_tensor):
    """Full K→0 reverse diffusion pass. Returns clean action chunk [chunk_size, act_dim]."""
    x = torch.randn(1, actor.chunk_size, actor.act_dim, device=DEVICE)
    scheduler.set_timesteps(scheduler.config.num_train_timesteps)

    with torch.no_grad():
        for k in scheduler.timesteps:
            k_tensor   = torch.tensor([k], dtype=torch.long, device=DEVICE)
            noise_pred = actor(obs_tensor, x, k_tensor)
            x = scheduler.step(noise_pred, k, x).prev_sample

    return x.squeeze(0).cpu().numpy()  # [chunk_size, act_dim]


def evaluate_actor(actor, scheduler, env, num_episodes, record_frames=False):
    """Runs num_episodes and returns (success_rate, frames)."""
    frames = []
    successes = 0

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done   = False
        ep_success = False

        while not done:
            obs_tensor   = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            action_chunk = run_denoising_chain(actor, scheduler, obs_tensor)

            for i in range(CHUNK_SIZE):
                obs, _, terminated, truncated, info = env.step(action_chunk[i])
                if record_frames:
                    frames.append(env.render())
                if info.get('success', False):
                    ep_success = True
                if terminated or truncated:
                    done = True
                    break

        if ep_success:
            successes += 1

    return successes / num_episodes, frames


def compare_bc_vs_dppo():
    """Evaluate BC checkpoint and best DPPO checkpoint; print success rate comparison."""
    import os

    scheduler = DDPMScheduler(
        num_train_timesteps=K,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    results = {}

    for label, ckpt in [("BC only", BC_CHECKPOINT), ("BC + DPPO", BEST_CHECKPOINT)]:
        if not os.path.exists(ckpt):
            print(f"  [{label}] Checkpoint not found: {ckpt} — skipping")
            continue
        print(f"\nEvaluating {label} ({ckpt}) over {NUM_EPISODES} episodes...")
        actor = load_actor(ckpt)
        env   = build_eval_env()
        rate, _ = evaluate_actor(actor, scheduler, env, NUM_EPISODES, record_frames=False)
        env.close()
        results[label] = rate
        print(f"  {label}: {rate*100:.1f}% success")

    if len(results) == 2:
        bc_rate   = results["BC only"]
        dppo_rate = results["BC + DPPO"]
        delta     = dppo_rate - bc_rate
        print("\n" + "="*40)
        print("  BC vs DPPO Comparison")
        print("="*40)
        print(f"  BC only   : {bc_rate*100:.1f}%")
        print(f"  BC + DPPO : {dppo_rate*100:.1f}%")
        print(f"  Δ (improvement) : {delta*100:+.1f}%")
        print("="*40)

        try:
            import matplotlib.pyplot as plt
            _, ax = plt.subplots(figsize=(5, 4))
            labels = list(results.keys())
            values = [results[l] * 100 for l in labels]
            bars = ax.bar(labels, values, color=["#4C72B0", "#DD8452"], width=0.4)
            ax.set_ylabel("Success Rate (%)")
            ax.set_title("BC vs DPPO — AdroitHandPen-v1")
            ax.set_ylim(0, 100)
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.1f}%", ha="center")
            plt.tight_layout()
            plt.savefig("bc_vs_dppo.png", dpi=150)
            print("  Plot saved → bc_vs_dppo.png")
        except ImportError:
            pass  # matplotlib optional


def render_video():
    """Record a video of the best available policy."""
    import os

    ckpt = BEST_CHECKPOINT if os.path.exists(BEST_CHECKPOINT) else BC_CHECKPOINT
    if not os.path.exists(ckpt):
        print(f"No checkpoint found at {BEST_CHECKPOINT} or {BC_CHECKPOINT}. Skipping video.")
        return

    print(f"\nRecording video using: {ckpt}")
    scheduler = DDPMScheduler(
        num_train_timesteps=K,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    actor = load_actor(ckpt)
    env   = build_eval_env()
    _, frames = evaluate_actor(actor, scheduler, env, num_episodes=3, record_frames=True)
    env.close()

    if frames:
        imageio.mimsave(OUTPUT_VIDEO, frames, fps=FPS)
        print(f"Video saved → {OUTPUT_VIDEO}")


if __name__ == "__main__":
    compare_bc_vs_dppo()
    render_video()
