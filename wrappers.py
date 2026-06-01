import gymnasium as gym
import numpy as np
class ActionChunkingWrapper(gym.Wrapper):
    """
    Executes a chunk of actions sequentially in the base environment.
    The policy predicts Tp actions, but we execute Ta actions open-loop.
    """
    def __init__(self, env, chunk_size:int):
        super().__init__(env)
        self.Ta=chunk_size

        #expand the action space to (Ta,action_dim)
        assert isinstance(env.action_space,gym.spaces.Box)
        self.single_action_dim=env.action_space.shape[0]
        #repeats the array actionspace(lower bound for each action dim)self.Ta times along the row
        low=np.tile(env.action_space.low,(self.Ta,1)) #(Ta,action_dim)
        high=np.tile(env.action_space.high,(self.Ta,1)) 
        #spaces.Box defines a continuous space for all action arrays between low and high
        #space.box is just a data structure in gym to define a space with low/upper bounds
        self.action_space=gym.spaces.Box(low=low,high=high,dtype=np.float32)

    def step(self,action_chunk):
        """
        executes Ta steps
        action chunk expected (Ta,action_dim)
        """
        total_reward=0.0
        done=False
        truncated=False
        info={}
        for i in range(self.Ta):
            obs,reward,done,truncated,step_info=self.env.step(action_chunk[i])
            total_reward+=reward
            info.update(step_info)

            if done or truncated:
                break

        return obs,total_reward,done,truncated,info
    

class DiffusionStateNormalizer(gym.ObservationWrapper):
    """
    Normalizes 1D state observations to a strict[-1,1] range
    Diffusion models degrade rapidly if state conditioning vectors explode
    """

    def __init__(self,env):
        super().__init__(env)
        assert isinstance(env.observation_space,gym.spaces.Box)
        self.eps=1e-8 # small number to prevent division by 0

        #mean variance to calculate the deviation
        self.running_mean=np.zeros(env.observation_space.shape,dtype=np.float32)#[obs_dim]
        self.running_var=np.ones(env.observation_space.shape,dtype=np.float32)#[obs_dim]
        self.count=1e-4

    def observation(self,observation):
        """
        This function is automatically called by Gym every time env.step() 
        or env.reset() produces a new observation.
        observation Shape: [obs_dim]
        """
        self.count+=1 # update total step
        
        #calculate how far this is from the mean
        delta=observation-self.running_mean
        #new mean=old mean+(x-oldmean)/new count
        self.running_mean+=delta/self.count # update the mean based on welford alg

        self.running_var+=delta*(observation-self.running_mean)#update the variance

        #calculate std
        var=self.running_var/self.count
        std=np.sqrt(np.maximum(var,self.eps))
        normalized_obs=(observation-self.running_mean)/std # z score

        return np.tanh(normalized_obs)

class RewardScaler(gym.RewardWrapper):
    """
    Scales rewards to maintain stable advantages for PPO.
    Dexterous manipulation tasks often have sparse or rapidly exploding dense rewards.
    """
    def __init__(self,env,scale:float=0.01):
        super().__init__(env)
        self.scale=scale

    def reward(self,reward):
        #called by gym after env.step()
        return reward*self.scale

def make_dppo_env(env_id:str,Ta:int,reward_scale:float=0.01):
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    env=gym.make(env_id)
    env=DiffusionStateNormalizer(env)
    env=RewardScaler(env,scale=reward_scale)
    env=ActionChunkingWrapper(env,chunk_size=Ta)
    env=gym.wrappers.RescaleAction(env,min_action=-1.0,max_action=1.0)
    return env




