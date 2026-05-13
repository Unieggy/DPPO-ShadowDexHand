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