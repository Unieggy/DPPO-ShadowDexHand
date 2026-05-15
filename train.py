import torch
import torch.nn as nn
import torch.optim as optim
from torch.utiles.data import DataLoader,TensorDataset

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
            batch_size=batch_states.to(device) #(batch_size,obs_dim)
            batch_actions=batch_actions.to(device)#(batch_size,chunk_size,act_dim)

            #1 randomize diffusion steps k
            #we train the network to denoise from any step in the sequence
            #random_ks:[batch_size]
            random_ks=torch.randin(0,K,(batch_size,),device=device,dtype=torch.long)


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

    for iteration in range(num_iterations):
        # EVENT 1: THE ROLLOUT (Gathering Data using the FROZEN Old Policy)
        buffer.clear()

        #collect a specific number of environment trajectories
        for t in range(buffer.total_capacity//K_prime):
            state,_=env.reset() # base env state
            state=torch.tensor(state,dtype=torch.float32,device=device).unsqueeze(0)

            #start diffusion from pure noise
            #noisy action:[1,chunk_size,act_dim]
            noisy_action=torch.randn((1,active_actor.chunk_size,active_actor.act_dim),device=device)

            #temporary storage for the K' fine tuning window
            window_states,window_actions,window_ks,windo_log_probs=[],[],[],[]

            with torch.no_grad(): #frozen model
                #run the reverse diffusion loop
                for k in reversed(range(K)):
                    #k_tensor [1]
                    k_tensor=torch.tensor([k],dtype=torch.long,device=device)

                    #old policy predicts noise for this specific k step
                    #noise_pred shape:[1,chunk_size,act_dim]
                    noise_pred=old_actor(state,noisy_action,k_tensor)

                    #get the mathmat