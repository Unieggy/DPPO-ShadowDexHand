import torch
import torch.nn as nn

class DashcamBuffer:
    """
    Records the internal (denoising steps) of the diffusion model 
    for the last K' active fine-tuning steps, alongside the environment rewards.
    """
    def __init__ (self,num_env_steps:int,K_prime:int,obs_dim:int,chunk_size:int,act_dim:int,device: str="cpu"):
        # num_env_steps: How many physical environment steps we take per rollout
        # K_prime: How many diffusion steps we are actively fine-tuning (e.g., the last 3 steps)
        self.total_capacity=num_env_steps*K_prime#total micro steps
        self.device=device
        self.ptr=0 # pointer to track where in the buffer


        #preallocate tensots for speed and memory efficiency
        self.states=torch.zeros((self.total_capacity,obs_dim),device=device)
        self.noisy_actions=torch.zeros((self.total_capacity,chunk_size,act_dim),device=device)
        self.k_steps=torch.zeros((self.total_capacity,),dtype=torch.long,device=device)
        self.log_probs=torch.zeros((self.total_capacity,),device=device)
        self.advantages=torch.zeros((self.total_capacity,),device=device)
        self.returns=torch.zeros((self.total_capacity,),device=device)

    def add_trajectory(self,states,noisy_actions,k_steps,log_probs,env_advantage,env_return):
        """
        Adds the K' micro-steps from ONE physical environment step into the buffer.
        Notice that env_advantage is a single scalar, but we assign it to all K' steps.
        
        Input Shapes (for a single environment step):
        - states: [K_prime, obs_dim]
        - noisy_actions: [K_prime, chunk_size, act_dim]
        - k_steps: [K_prime]
        - log_probs: [K_prime]
        - env_advantage: [Scalar]
        - env_return: [Scalar]
        """
        K_prime=states.shape[0]

        #determine the slice in our pre allocated tensors
        idx=slice(self.ptr,self.ptr+K_prime) # start,ending index

        self.states[idx]=states
        self.noisy_actions[idx]=noisy_actions
        self.k_steps[idx]=k_steps
        self.log_probs[idx]=log_probs

        #broadcasting, copy the single physical advatange/return to all K' internal steps.
        self.advantages[idx]=env_advantage.expand(K_prime)
        self.returns[idx]=env_return.expand(K_prime)

        self.ptr+=K_prime

    def get_all(self):
        #returns the full buffer for the ppo update

        #normalize advantages across the whole batch to stablize ppo gradients
        adv=self.advantages
        normalized_adv=(adv-adv.mean())/(adv.std()+1e-8)

        return(self.states,self.noisy_actions,self.k_steps,self.log_probs,normalized_adv,self.returns)
    
    def clear(self):
        self.ptr=0

def calculate_gaussian_log_prob(predicted_noise, target_noise, log_variance):
    """
    Calculates the Log-Likelihood of the diffusion step transition.
    In DDIM, the reverse step is treated as a Gaussian distribution.
    
    predicted_noise Shape: [batch, chunk_size, act_dim] (Output of our MLP Actor)
    target_noise Shape: [batch, chunk_size, act_dim] (The actual noise we added during training)
    log_variance Shape: [batch, 1, 1] (From the diffusion scheduler)
    """
    # Math: log P(x) = -0.5 * ( (x - mu)^2 / var + log(var) + log(2*pi) )
    # Because diffusion predicts NOISE instead of the data directly,
    # we calculate the log prob of the noise matching.

    #calculate mean squared error between predicted and actual noise
    #shape[batch,chunk_size,act_dim]
    squared_error=(predicted_noise-target_noise)**2

    #divide by variance
    scaled_error=squared_error*torch.exp(-log_variance)

    #sum the log likelihoods across chunk and action dimension
    #so a single scalar log_prob per batch
    #log(2*pi) approx 1.837877
    #outputshape:[batch]

    log_prob=-0.5*(scaled_error+log_variance+1.837877).sum(dim=(1,2))

    return log_prob