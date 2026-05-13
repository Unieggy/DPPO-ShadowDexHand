import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    """
    Standard sinusoidal positional embedding to encode the diffusion time step-k
    diffusion models must know which step of noise they are predicting
    """

    def __init__(self,dim:int):
        super().__init__()
        self.dim=dim #target embedding dimension

    def forward(self,x):
        """
        x is the step k in size(batch_size)
        sinusoidal equation
        Mathematical definition of the Sinusoidal Positional Embeddings:
        
        PE(k, 2i)   = sin( k * (1 / 10000^(2i/d)) )
        PE(k, 2i+1) = cos( k * (1 / 10000^(2i/d)) )
        
        Where:
        - k is the diffusion time-step.
        - d is the total embedding dimension.
        - i is the frequency index (ranging from 0 to d/2 - 1).
        """

        device=x.device
        # we only need half so we can concat the sin and cos together
        # sin only deal with the even index and cos only deal with the odd
        half_dim=self.dim//2

        #calculate the frequencies for sin/cos wave
        #shape [half_dim]
        #Math:ln(10000)/(d/2-1)
        #torch.arange generates i values from 0 to half_dim-1 [0....half_dim-1]
        #Math exp(-i*ln(10000/(d/2-1))) view d/2-1 d/2 since we start at 0 which turns into 
        #exp(-2i/d(ln10000)) which turns into 1/10000^2i/d

        emb=math.log(10000)/(half_dim-1)
        emb=torch.exp(torch.arange(half_dim,device=device)*-emb)

        #multiply time step k by frequencies
        #x[:,None] shape:[batch_size,1]
        #emb[None,:] shape[1,half_dim]
        #output shape:[batch_size,half_dim]
        #Math k*1/10000^2i/d for every i
        emb=x[:,None]*emb[None,:]

        #concat sine and cosine transformation
        # output shape: (batch_size,dim)
        #[ sin(k*f0), sin(k*f1), ..., sin(k*f_{H-1}),
        #cos(k*f0), cos(k*f1), ..., cos(k*f_{H-1}) ]

        emb=torch.cat((emb.sin(),emb.cos()),dim=-1)
        return emb
    
class DiffusionMLPActor(nn.Module):
    """
    The Actor: Predicts the noise added to the action chunk at step k
    """
    def __init__(self,obs_dim:int,act_dim:int,chunk_size:int,hidden_dim:int=256):
        super().__init__()
        self.chunk_size=chunk_size #Tp total predicted steps
        self.act_dim=act_dim # action dimension for each steps

        #flatten the action chunk for the MLP
        self.flat_action_dim=self.chunk_size*self.act_dim

        #time step embedding network
        self.time_mlp=nn.Sequential(
            SinusoidalPosEmb(hidden_dim), #encodes each k steps to [batch_size,hidden_dim]
            nn.Linear(hidden_dim,hidden_dim*2),
            nn.Mish(),
            nn.Linear(hidden_dim*2,hidden_dim)
        )

        #input dimension:State+flattened action+time embedding
        input_dim=obs_dim+self.flat_action_dim+hidden_dim

        #the core mlp backbone
        #Mish is preferred over ReLu in diffusion for smoother gradients
        self.net=nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim,self.flat_action_dim)# output matches the flattened action dim
        )

    def forward(self,state,noisty_action_chunk,k_step):
        """
        state Shape: [batch_size, obs_dim]
        noisy_action_chunk Shape: [batch_size, chunk_size, act_dim]
        k_step Shape: [batch_size]
        """

        #1, Flatten the 2d action chunk into a 1d vector per batch
        #shape:[batch_size,chunk_size*action_dim]
        flat_action=noisty_action_chunk.view(-1,self.flat_action_dim)

        #2 embed the diffusion step
        #shape[batch_size,hidden_dim]
        #pass the k for eg 42 into the sinusoidal pos emb
        #which converts the integer to a vector of [1,256]
        time_emb=self.time_mlp(k_step)

        #3 concat all inputs along the feature dimension
        #shape[batch_size,obs_dim+flat_action_dim+hidden_dim]
        x=torch.cat([state,flat_action,time_emb],dim=-1)

        # 4 predict the noise
        #shape[batch_size,chunk_size*act_dim]
        predicted_flat_noise=self.net(x)

        #5 reshape back to the original action chunk size
        #shape[batch_size,chunk_size,act_dim]
        predicted_noise=predicted_flat_noise.view(-1,self.chunk_size,self.act_dim)

        return predicted_noise
    
class ValueCritic(nn.Module):
    """
    The Critic: Predicts the scalar value V(s) of a given state.
    Used exclusively to calculate the Generalized Advantage Estimate (GAE).
    """
    def __init__(self,obs_dim:int,hidden_dim:int=256):
        super().__init__()

        #standard mlp, just map state to a single value
        self.net=nn.Sequential(
            nn.Linear(obs_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim,1)# output is a single value per batch/k step
        )
    def forward(self,state):
        #state shape:[batch_size,obs_dim]
        return self.net(state)