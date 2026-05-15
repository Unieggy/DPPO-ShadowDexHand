import os
import torch
from copy import deepcopy
from diffusers import DDPMScheduler

from network import DiffusionMLPActor, ValueCritic
from dppo_math import DashcamBuffer
from train import train_behavior_cloning, train_dppo
from wrappers import make_dppo_env

# ── Config ────────────────────────────────────────────────────────────────────
ENV_ID      = "HandManipulateBlockRotateXYZ-v1"
OBS_DIM     = 75    # 61 joint obs + 7 achieved_goal + 7 desired_goal (after flatten)
ACT_DIM     = 20    # Shadow Hand has 20 actuated DoF
CHUNK_SIZE  = 4     # Tp: predict this many future actions
EXECUTE_TA  = 2     # Ta: how many of the chunk to actually execute open-loop
K           = 100   # total diffusion steps
K_PRIME     = 3     # fine-tune only the last K' denoising steps
NUM_ENV_STEPS = 64  # env steps collected per rollout
HIDDEN_DIM  = 256

BC_EPOCHS     = 50
BC_BATCH_SIZE = 256

SAVE_DIR = "checkpoints"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Running on: {DEVICE}")

    # ── Environment ──────────────────────────────────────────────────────────
    env = make_dppo_env(ENV_ID, Ta=EXECUTE_TA)

    # ── Networks ─────────────────────────────────────────────────────────────
    actor  = DiffusionMLPActor(OBS_DIM, ACT_DIM, CHUNK_SIZE, HIDDEN_DIM).to(DEVICE)
    critic = ValueCritic(OBS_DIM, HIDDEN_DIM).to(DEVICE)

    # ── Diffusion scheduler ───────────────────────────────────────────────────
    # squaredcos_cap_v2 is the schedule used in the original DPPO paper
    scheduler = DDPMScheduler(
        num_train_timesteps=K,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )

    # ── Rollout buffer ────────────────────────────────────────────────────────
    buffer = DashcamBuffer(
        num_env_steps=NUM_ENV_STEPS,
        K_prime=K_PRIME,
        obs_dim=OBS_DIM,
        chunk_size=CHUNK_SIZE,
        act_dim=ACT_DIM,
        device=DEVICE,
    )

    # ── Phase 1: Behavior Cloning ─────────────────────────────────────────────
    # Expects a file expert_data.pt with keys "states" and "action_chunks".
    # If not present, DPPO starts from a randomly initialized policy (slower to converge).
    expert_data_path = "expert_data.pt"
    if os.path.exists(expert_data_path):
        data = torch.load(expert_data_path, map_location=DEVICE)
        expert_states  = data["states"]         # [N, OBS_DIM]
        expert_chunks  = data["action_chunks"]  # [N, CHUNK_SIZE, ACT_DIM]
        print(f"Loaded {len(expert_states)} expert transitions. Starting BC pretraining...")
        actor = train_behavior_cloning(
            actor, scheduler, expert_states, expert_chunks,
            K=K, epochs=BC_EPOCHS, batch_size=BC_BATCH_SIZE, device=DEVICE,
        )
        torch.save(actor.state_dict(), os.path.join(SAVE_DIR, "actor_bc.pt"))
        print("BC pretraining done.")
    else:
        print("No expert_data.pt found — skipping BC, starting DPPO from random init.")

    # ── Phase 2: DPPO ─────────────────────────────────────────────────────────
    old_actor = deepcopy(actor).to(DEVICE)
    print("Starting DPPO fine-tuning...")
    train_dppo(env, old_actor, actor, critic, buffer, scheduler, K, K_PRIME, device=DEVICE)

    # ── Save final checkpoints ────────────────────────────────────────────────
    torch.save(actor.state_dict(),  os.path.join(SAVE_DIR, "actor_final.pt"))
    torch.save(critic.state_dict(), os.path.join(SAVE_DIR, "critic_final.pt"))
    print(f"Training complete. Checkpoints saved to {SAVE_DIR}/")

    env.close()


if __name__ == "__main__":
    main()
