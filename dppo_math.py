import torch
import torch.nn as nn

class DashcamBuffer:
    """
    Records the internal (denoising steps) of the diffusion model 
    for the last K' active fine-tuning steps, alongside the environment rewards.
    """
