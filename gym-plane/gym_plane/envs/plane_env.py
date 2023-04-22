import gym
import numpy as np
from plane_toolkit.dynamics.plane_dynamics import normalized_rk4_step

class PlaneEnv(gym.Env):

    def __init__(self, info_dict):
        super(PlaneEnv, self).__init__()
    
        self.info_dict = info_dict
        self.h = info_dict["h"]
        self.target_traj = info_dict["target_traj"]
        assert self.target_traj.shape[1] == 13
        self.num_prev_steps = info_dict["num_prev_steps"]
        self.num_future_steps = info_dict["num_future_steps"]
        self.clip_observations = info_dict["clip_observations"]
        self.clip_actions = info_dict["clip_actions"]
        self.pos_rew_scale = info_dict["pos_rew_scale"]
        self.quat_rew_scale = info_dict["quat_rew_scale"]
        self.is_debug = info_dict["is_debug"]

        #observation space is
        #13 for xaction
        #16*num_prev_steps for previous poses and controls
        #8*num_future_steps for next desired poses and if valid
        self.observation_shape = (13 + 16*self.num_prev_steps + 8*self.num_future_steps,)
        self.observation_space = gym.spaces.Box(
            low=-self.clip_observations, high=self.clip_observations, shape=self.observation_shape, dtype=np.float32,
        )

        #action space is control vector
        self.action_shape = (3,)
        self.action_space = gym.spaces.Box(
        low=-self.clip_actions, high=self.clip_actions, shape=self.action_shape, dtype=np.float32,
        )

        self.N = self.target_traj.shape[0]
        if self.is_debug:
            self.N = 101

    def reset(self):
        if self.is_debug:
            self.x = np.array([-10.0, 0.0, 10.0, 0.0, 0.9891199724086334, 
                           0.0, -0.1471111150876924, 20.0,
                           0.0, 0.5678625279339364, 0.0, 0.0, 0.0])
        else:
            #TODO confirm starting traj is start state
            self.x = np.copy(self.target_traj[0, :])
        
        #all prevs xs are 0
        self.prev_xs = np.zeros((self.num_prev_steps, 13))
        #all prev controls are 0
        self.prev_us = np.zeros((self.num_prev_steps, 3))

        #read future steps from target trajectories
        self.future_steps = np.zeros((self.num_future_steps, 8))
        max_future_ind = np.min([self.num_future_steps, self.target_traj.shape[0] - 1])
        for i in range(max_future_ind):
            self.future_steps[i, 0:7] = self.target_traj[1 + i, 0:7]
            self.future_steps[i, 7] = 1.0

        self.num_steps = 0

        self._make_observation()

        return self.obs

    def step(self, action):
        if self.is_debug:
            action = np.array([0.0, 0.27090666435590915, 0.0])

        #u is action
        self.u = np.clip(action, -self.clip_actions, self.clip_actions)

        #first update previous xs
        self.prev_xs[1:, :] = self.prev_xs[0:-1, :]
        self.prev_xs[0, :] = self.x[:]
        
        #now update prev us
        self.prev_us[1:, :] = self.prev_us[0:-1, :]
        self.prev_us[0, :] = self.u[:]

        #now update future steps
        self.future_steps[0:-1, :] = self.future_steps[1:, :]
        future_ind = self.num_steps + 1 + self.num_future_steps
        if future_ind < self.target_traj.shape[0]:
            self.future_steps[-1, 0:7] = self.target_traj[future_ind, 0:7]
            self.future_steps[-1, 7] = 1.0
        else:
            self.future_steps[-1, 0:7] = 0.0
            self.future_steps[-1, 7] = 0.0

        self.x = normalized_rk4_step(self.x, self.u, self.h)
        self.x = np.clip(self.x, -self.clip_observations, self.clip_observations)

        pos_error = np.linalg.norm(self.x[0:3] - self.target_traj[self.num_steps + 1, 0:3])
        quat_error = np.linalg.norm(self.x[3:7] - self.target_traj[self.num_steps + 1, 3:7])
        
        pos_rew = 1.0/(1 + pos_error**2)
        pos_rew *= self.pos_rew_scale

        quat_rew = 1.0/(1 + quat_error**2)
        quat_rew *= self.quat_rew_scale

        self.num_steps += 1

        self._make_observation()

        done = self.num_steps >= self.N - 1
        obs = self.obs
        info = {}
        reward = pos_rew + quat_rew

        return obs, reward, done, info
  
    def render(self):
        pass

    def _make_observation(self):
        self.obs = np.concatenate((self.x, self.prev_xs.flatten(),
                                      self.prev_us.flatten(), 
                                      self.future_steps.flatten())).astype(np.float32)