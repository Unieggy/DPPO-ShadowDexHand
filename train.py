import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset
from dppo_math import calculate_gaussian_log_prob, compute_ppo_objective, get_ddpm_log_variance

def train_behavior_cloning(actor,diffusion_scheduler,expert_states,expert_action_chunks,K:int,epochs:int,batch_size:int,device="cpu"):
    """
    Supervised learning pretraining to train the actor to mimic expert demonstrations before RL
    Uses a ddpm's forward noise to jump instantly to random diffusion steps k 
    and corrupt perfect human data with Gaussian noise. 
    The network learns to guess the noise added
    """

    actor.train()
    optimizer=optim.AdamW(actor.parameters(),lr=1e-4)
    loss_fn=nn.MSELoss()

    #pack expert data into pytorch dataloader
    #expert state:[total_samples,obs_dim]
    #expert_action_chunks:[total_samples,chunk_size,act_dim]
    dataset=TensorDataset(expert_states,expert_action_chunks)
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True)

    for epoch in range(epochs):
        epoch_loss=0.0
        for batch_states,batch_actions in dataloader:
            batch_states=batch_states.to(device) #(batch_size,obs_dim)
            batch_actions=batch_actions.to(device)#(batch_size,chunk_size,act_dim)

            #1 randomize diffusion steps k
            #we train the network to denoise from any step in the sequence
            #random_ks:[batch_size]
            random_ks=torch.randint(0,K,(batch_states.shape[0],),device=device,dtype=torch.long)


            #2 generate pure gaussian noise
            #pure noise:[batch_size,chunk_size,act_dim]
            pure_noise=torch.randn_like(batch_actions)

            #3 create the trianing target
            #mathmatically corruput the expert action with the pure noise to simulate step k
            #noist_actions :[batch_size,chunk_size,act_dim]
            #hugging face diffuser, not a neural network just a strict mathmatical formulas to do the noise adding
            #built in method to do forward diffusion, actor is our diffusion mlp defined in network

            noisy_actions=diffusion_scheduler.add_noise(batch_actions,pure_noise,random_ks)

            #4 neural network forward pass
            #actor predicts what noise was added to the clean action
            #predicted_noise: [batch_size,chunk_size,act_dim]
            predicted_noise=actor(batch_states,noisy_actions,random_ks)

            # 5 mse loss
            loss=loss_fn(predicted_noise,pure_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()

        if (epoch+1)%10==0 or epoch==0:
            print(f"  BC Epoch {epoch+1}/{epochs}  loss={epoch_loss/len(dataloader):.4f}")
    return actor

def evaluate_policy(env, actor, diffusion_scheduler, K, device, num_episodes=5):
    """
    Runs deterministic evaluation episodes — no exploration noise.
    The actor uses its mean prediction (noise_pred) directly at every denoising step.
    Returns the success rate across num_episodes.
    """
    actor.eval()
    successes=0
    Ta=env.action_space.shape[0]

    with torch.no_grad():
        for ep in range(num_episodes):
            obs,_=env.reset()
            done=False
            ep_success=False

            while not done:
                obs_tensor=torch.tensor(obs,dtype=torch.float32,device=device).unsqueeze(0)

                # deterministic K→0 denoising: no sigma*z added
                x=torch.randn(1,actor.chunk_size,actor.act_dim,device=device)
                for k in reversed(range(K)):
                    k_tensor=torch.tensor([k],dtype=torch.long,device=device)
                    noise_pred=actor(obs_tensor,x,k_tensor)
                    x=diffusion_scheduler.step(noise_pred,k,x).prev_sample

                action=x.cpu().numpy()[0,:Ta]
                obs,_,terminated,truncated,info=env.step(action)
                if info.get('success',False):
                    ep_success=True
                done=terminated or truncated

            if ep_success:
                successes+=1

    actor.train()
    return successes/num_episodes


def train_dppo(env,old_actor,active_actor,critic, buffer, diffusion_scheduler,K,K_prime,device="cpu",save_dir="checkpoints",save_every=50,eval_every=20,num_eval_episodes=5):
    """
    PROCESS: REINFORCEMENT LEARNING (DPPO)
    Fine-tunes the behavior-cloned policy using environment rewards.
    """

    optimizer_actor=optim.AdamW(active_actor.parameters(),lr=1e-5)
    optimizer_critic=optim.AdamW(critic.parameters(),lr=3e-4)

    num_iterations=1000
    ppo_epochs=4 #times we sweep thru the buffer per iteration
    num_env_steps=buffer.total_capacity//K_prime
    best_success_rate=-1.0

    for iteration in range(num_iterations):
        # EVENT 1: THE ROLLOUT (Gathering Data using the FROZEN Old Policy)
        buffer.clear()
        rollout_rewards=[]

        print(f"\n{'='*50}")
        print(f"Iteration {iteration+1}/{num_iterations}")
        print(f"  [Rollout] Collecting {num_env_steps} environment steps...")

        #collect a specific number of environment steps across episodes
        obs,_=env.reset()
        Ta=env.action_space.shape[0]

        for t in range(num_env_steps):

            #[1, obs_dim]
            state=torch.tensor(obs,dtype=torch.float32,device=device).unsqueeze(0)

            #start diffusion from pure noise
            #noisy action:[1,chunk_size,act_dim]
            noisy_action=torch.randn((1,active_actor.chunk_size,active_actor.act_dim),device=device)

            #temporary storage for the K' fine tuning window
            window_states,window_actions,window_committed_noises,window_ks,window_log_probs=[],[],[],[],[]

            with torch.no_grad(): #frozen model
                #run the reverse diffusion loop
                for k in reversed(range(K)):
                    #k_tensor [1]
                    k_tensor=torch.tensor([k],dtype=torch.long,device=device)

                    #old policy predicts noise for this specific k step
                    #noise_pred shape:[1,chunk_size,act_dim]
                    noise_pred=old_actor(state,noisy_action,k_tensor)

                    # sample committed noise: policy mean + scheduler variance
                    # shape: [1,1,1] → broadcasts over [1,chunk_size,act_dim]
                    log_var=get_ddpm_log_variance(diffusion_scheduler,k_tensor,device)
                    sigma_k=torch.exp(0.5*log_var)
                    committed_noise=noise_pred+sigma_k*torch.randn_like(noise_pred)

                    if k<K_prime:
                        old_log_prob=calculate_gaussian_log_prob(noise_pred,committed_noise,log_var)
                        window_states.append(state.squeeze(0))
                        window_actions.append(noisy_action.squeeze(0))
                        window_committed_noises.append(committed_noise.squeeze(0))
                        window_ks.append(k_tensor.squeeze(0))
                        window_log_probs.append(old_log_prob.squeeze(0))

                    noisy_action=diffusion_scheduler.step(noise_pred,k,noisy_action).prev_sample

                #env execution
                final_action_chunk=noisy_action.cpu().numpy()[0,:Ta]
                obs,reward,done,truncated,_=env.step(final_action_chunk)
                rollout_rewards.append(reward)

                #advantage calculation and broadcast
                state_value=critic(state).squeeze()
                next_state=torch.tensor(obs,dtype=torch.float32,device=device).unsqueeze(0)
                gamma=0.99
                next_val=critic(next_state).squeeze()*(1.0-float(done or truncated))
                env_return=torch.tensor([reward],device=device)+gamma*next_val.detach()
                env_advantage=env_return-state_value

                buffer.add_trajectory(
                    torch.stack(window_states),torch.stack(window_actions),
                    torch.stack(window_committed_noises),torch.stack(window_ks),
                    torch.stack(window_log_probs),env_advantage,env_return
                )

                if done or truncated:
                    obs,_=env.reset()

        mean_reward=sum(rollout_rewards)/len(rollout_rewards)
        print(f"  [Rollout] Done. Mean reward: {mean_reward:.4f}  Min: {min(rollout_rewards):.4f}  Max: {max(rollout_rewards):.4f}")

        #PARTB the ppo update
        print(f"  [PPO]     Updating policy over {ppo_epochs} epochs...")

        b_states,b_noisy_actions,b_committed_noises,b_k_steps,b_old_log_probs,b_advs,b_rets=buffer.get_all()

        for epoch in range(ppo_epochs): #PPO epochs
            #critic update
            value=critic(b_states).squeeze()
            critic_loss=torch.nn.functional.mse_loss(value,b_rets)
            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()

            #actor update(DPPO )
            new_noise_pred=active_actor(b_states,b_noisy_actions,b_k_steps)
            log_variance=get_ddpm_log_variance(diffusion_scheduler,b_k_steps,device)
            new_log_probs=calculate_gaussian_log_prob(new_noise_pred,b_committed_noises,log_variance)

            actor_loss=compute_ppo_objective(new_log_probs,b_old_log_probs,b_advs)

            optimizer_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(active_actor.parameters(), 0.5)
            optimizer_actor.step()

            print(f"    Epoch {epoch+1}/{ppo_epochs} — actor_loss: {actor_loss.item():.4f}  critic_loss: {critic_loss.item():.4f}")

        old_actor.load_state_dict(active_actor.state_dict())
        print(f"  [Policy]  Old actor synced with updated weights.")

        if (iteration+1) % save_every == 0:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(active_actor.state_dict(), os.path.join(save_dir, f"actor_iter{iteration+1}.pt"))
            torch.save(critic.state_dict(),       os.path.join(save_dir, f"critic_iter{iteration+1}.pt"))
            print(f"  [Saved]   Checkpoint saved at iteration {iteration+1} → {save_dir}/")

        if (iteration+1) % eval_every == 0:
            print(f"  [Eval]    Running {num_eval_episodes} deterministic episodes...")
            success_rate=evaluate_policy(env,active_actor,diffusion_scheduler,K,device,num_eval_episodes)
            print(f"  [Eval]    Success rate: {success_rate*100:.1f}%  (best so far: {best_success_rate*100:.1f}%)")
            if success_rate>best_success_rate:
                best_success_rate=success_rate
                os.makedirs(save_dir, exist_ok=True)
                torch.save(active_actor.state_dict(), os.path.join(save_dir, "actor_best.pt"))
                print(f"  [Best]    New best! actor_best.pt saved (success rate: {success_rate*100:.1f}%)")


