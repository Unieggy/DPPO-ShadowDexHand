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
        for batch_states,batch_actions in dataloader:
            batch_states=batch_states.to(device) #(batch_size,obs_dim)
            batch_actions=batch_actions.to(device)#(batch_size,chunk_size,act_dim)

            #1 randomize diffusion steps k
            #we train the network to denoise from any step in the sequence
            #random_ks:[batch_size]
            random_ks=torch.randint(0,K,(batch_size,),device=device,dtype=torch.long)


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
    return actor

def train_dppo(env,old_actor,active_actor,critic, buffer, diffusion_scheduler,K,K_prime,device="cpu"):
    """
    PROCESS: REINFORCEMENT LEARNING (DPPO)
    Fine-tunes the behavior-cloned policy using environment rewards.
    """

    optimizer_actor=optim.AdamW(active_actor.parameters(),lr=1e-5)
    optimizer_critic=optim.AdamW(critic.parameters(),lr=3e-4)

    num_iterations=1000
    ppo_epochs=4 #times we sweep thru the buffer per iteration
    num_env_steps=buffer.total_capacity//K_prime

    for iteration in range(num_iterations):
        # EVENT 1: THE ROLLOUT (Gathering Data using the FROZEN Old Policy)
        buffer.clear()
        rollout_rewards=[]

        print(f"\n{'='*50}")
        print(f"Iteration {iteration+1}/{num_iterations}")
        print(f"  [Rollout] Collecting {num_env_steps} environment steps...")

        #collect a specific number of environment trajectories
        for t in range(num_env_steps):
            
            state,_=env.reset() # base env state

            #[1, obs_dim]
            state=torch.tensor(state,dtype=torch.float32,device=device).unsqueeze(0)

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
                Ta=env.action_space.shape[0]
                final_action_chunk=noisy_action.cpu().numpy()[0,:Ta]
                next_state,reward,done,_,_=env.step(final_action_chunk)

                #advtange calculation and broadcast
                state_value=critic(state).squeeze()
                env_advantage=torch.tensor([reward],device=device)-state_value
                env_return=torch.tensor([reward],device=device)

                buffer.add_trajectory(
                    torch.stack(window_states),torch.stack(window_actions),
                    torch.stack(window_committed_noises),torch.stack(window_ks),
                    torch.stack(window_log_probs),env_advantage,env_return
                )

        #PARTB the ppo update

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
            optimizer_actor.step()

        old_actor.load_state_dict(active_actor.state_dict())


