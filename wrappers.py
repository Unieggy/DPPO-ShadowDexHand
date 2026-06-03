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
            # preserve success=True once achieved; don't let a later sub-step overwrite it
            prev_success = info.get('success', False)
            info.update(step_info)
            if prev_success:
                info['success'] = True

            if done or truncated:
                break

        return obs,total_reward,done,truncated,info
    

class DiffusionStateNormalizer(gym.ObservationWrapper):
    """
    Normalizes 1D state observations to [-1,1] via tanh(z-score).
    Accepts fixed mean/std from expert data (preferred) or falls back to
    Welford online estimation if none are provided.
    """

    def __init__(self, env, mean=None, std=None):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)
        self.eps = 1e-8

        if mean is not None and std is not None:
            self.fixed = True
            self.mean = np.array(mean, dtype=np.float32)
            self.std  = np.array(std,  dtype=np.float32)
        else:
            self.fixed = False
            self.running_mean = np.zeros(env.observation_space.shape, dtype=np.float32)
            self.running_var  = np.ones(env.observation_space.shape,  dtype=np.float32)
            self.count = 1e-4

    def observation(self, observation):
        if self.fixed:
            return np.tanh((observation - self.mean) / (self.std + self.eps)).astype(np.float32)

        self.count += 1
        delta = observation - self.running_mean
        self.running_mean += delta / self.count
        self.running_var  += delta * (observation - self.running_mean)
        std = np.sqrt(np.maximum(self.running_var / self.count, self.eps))
        return np.tanh((observation - self.running_mean) / std)

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

def make_dppo_env(env_id:str, Ta:int, reward_scale:float=0.1, obs_mean=None, obs_std=None):
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    env=gym.make(env_id)
    env=DiffusionStateNormalizer(env, mean=obs_mean, std=obs_std)
    env=RewardScaler(env,scale=reward_scale)
    env=ActionChunkingWrapper(env,chunk_size=Ta)
    env=gym.wrappers.RescaleAction(env,min_action=-1.0,max_action=1.0)
    return env




