import os
import gym
import numpy as np
from plane_toolkit.dynamics.plane_dynamics import normalized_rk4_step, calc_quat_error, YakPlane 
from plane_toolkit.utils import io

class PlaneEnv(gym.Env):

    def __init__(self, info_dict):
        super(PlaneEnv, self).__init__()
    
        ###parse info_dict
        self.info_dict = info_dict
    
        #target trajectories
        target_traj_yml = info_dict["target_traj_yml"]
        self.load_target_trajectories(target_traj_yml)

        #max/min observation space scalings from input
        self.obs_bound_factor = info_dict["obs_bound_factor"]

        self.pos_rew_scale = info_dict["pos_rew_scale"]
        self.quat_rew_scale = info_dict["quat_rew_scale"]
        self.u_rew_scale = info_dict["u_rew_scale"]
        ###end parse info_dict

        #for means and stds, use opt control output
        #seems more relatable than interpolated waypoints
        self.pos_mean = self.ilqr_res[:, 0:3].mean()
        self.pos_std = self.ilqr_res[:, 0:3].std()
        self.v_mean = self.ilqr_res[:, 7:10].mean()
        self.v_std = self.ilqr_res[:, 7:10].std()
        self.w_mean = self.ilqr_res[:, 10:13].mean()
        self.w_std = self.ilqr_res[:, 10:13].std()

        #dynamics model requires actions to be in (0:255)
        self.action_min = 0.0
        self.action_max = 255.0

        self.obs_bound = self.obs_bound_factor*np.max(np.abs(self.ilqr_res))

        #for starters observation space will
        #13 for x
        #4 for last control vector
        #1 for timestep

        # self.observation_shape = (18,)
        # self.observation_space = gym.spaces.Box(
        #     low=-self.obs_bound, high=self.obs_bound, shape=self.observation_shape, dtype=np.float32,
        # )
        # self.obs = np.zeros(self.observation_shape, dtype=np.float32)
        self.observation_shape = (4,)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=self.observation_shape, dtype=np.float32,
        )
        self.obs = np.zeros(self.observation_shape, dtype=np.float32)

        #action space is control vector
        self.action_shape = (4,)
        self.action_space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=self.action_shape, dtype=np.float32,
        )

        self.N = self.ilqr_control.shape[0]
        self.yak = YakPlane() 

    def load_target_trajectories(self, target_traj_yml):
        target_traj_info = io.read_yaml(target_traj_yml)

        self.Tf = target_traj_info["Tf"]
        self.h = target_traj_info["h"]

        dirname = os.path.dirname(target_traj_yml)

        ilqr_ref_path = os.path.join(dirname, target_traj_info["ilqr_ref"])
        ilqr_res_path = os.path.join(dirname, target_traj_info["ilqr_res"])
        ilqr_control_path = os.path.join(dirname, target_traj_info["ilqr_control"])

        ilqr_ref = io.read_csv(ilqr_ref_path)
        ilqr_res = io.read_csv(ilqr_res_path)
        ilqr_control = io.read_csv(ilqr_control_path)

        if not (ilqr_ref.shape == ilqr_res.shape):
            raise RuntimeError("ilqr ref and res shapes do not match")
        
        if not (ilqr_ref.shape[0] - 1 == (ilqr_control.shape[0])):
            raise RuntimeError("ilqr ref and control shapes are inconsistent")
        
        self.ilqr_ref = ilqr_ref
        self.ilqr_res = ilqr_res
        self.ilqr_control = ilqr_control

    def reset(self):
        #initial state

        self.x = np.copy(self.ilqr_res[0, :])
        #TODO get values that lead to 0 control
        self.last_u = np.zeros((4,))
        self.num_steps = 0

        self._make_observation(False)

        return np.copy(self.obs)

    def step(self, action):
        if not (len(action.shape) == 1):
            raise RuntimeError("dim issue again")

        #rescale action to get us
        self.last_u = self.unnormalize_action(action)

        #update x
        x_new = normalized_rk4_step(self.yak, self.x, self.last_u, self.h)

        if not len(x_new.shape) == 1:
            raise RuntimeError("dim issue again")
        
        if (np.max(np.abs(x_new)) >= self.obs_bound):
            done = 1
            reward = 0
            self.x = x_new
            self.num_steps += 1
            self._make_observation(done, bad_obs=True)
            obs = np.copy(self.obs)
            self.latest_pos_error = 100
            self.latest_quat_error = 100
            self.latest_u_error = 100
            info = {}
            return obs, reward, done, info

        if np.isnan(x_new).sum():
            raise RuntimeError('x_new isnan')

        #compare to target pos which is num_steps + 1
        reward, pos_error, quat_error, u_error = self.compute_reward(x_new[0:3], x_new[3:7], self.last_u)

        self.x = x_new
        self.num_steps += 1
        done = self.num_steps >= self.N

        self._make_observation(done)

        obs = np.copy(self.obs)
        info = {}

        self.latest_pos_error = pos_error
        self.latest_quat_error = quat_error
        self.latest_u_error = u_error

        return obs, reward, done, info
  
    def render(self):
        pass

    def _make_observation(self, done, bad_obs=False):
        # self.obs[0:3] = (self.x[0:3] - self.pos_mean) / self.pos_std
        # self.obs[7:10] = (self.x[7:10] - self.v_mean) / self.v_std
        # self.obs[10:13] = (self.x[10:] - self.w_mean) / self.w_std
        # self.obs[13:17] = self.normalize_action(self.last_u)
        # self.obs[17] = 2*(self.num_steps / self.N) - 1
        if not done:
            self.obs[:] = self.normalize_action(self.ilqr_control[self.num_steps, :])
        else:
            pass

        #should never be needed I think but adding just in case
        #self.obs = np.clip(self.obs, -self.obs_bound, self.obs_bound)
        if not bad_obs and np.max(np.abs(self.obs)) >= self.obs_bound:
            raise RuntimeError("obs_bound reached in make_observation for step: " + str(self.num_steps))
        
    def normalize_action(self, u):
        return 2*(u - self.action_min) / (self.action_max - self.action_min) - 1
    
    def unnormalize_action(self, action):
        u = ((action + 1) / 2) * (self.action_max - self.action_min) + self.action_min
        return u
    
    def compute_reward(self, pos, quat, u, single_axis=True):
        if single_axis:
            pos_error = np.linalg.norm(pos - self.ilqr_res[self.num_steps + 1, 0:3])
            quat_error = calc_quat_error(quat, self.ilqr_res[self.num_steps + 1, 3:7])
            u_error = np.linalg.norm(u - self.ilqr_control[self.num_steps])
        else:
            pos_error = np.linalg.norm(pos - self.ilqr_res[self.num_steps + 1, 0:3], axis=1)
            
            quat_error = np.zeros(pos_error.shape)
            for i in range(quat_error.shape[0]):
                quat_error[i] = calc_quat_error(quat[i], self.ilqr_res[self.num_steps + 1, 3:7])
            u_error = np.linalg.norm(u - self.ilqr_control[self.num_steps], axis=1)

        pos_rew = 1.0/(1 + pos_error**2)
        pos_rew *= self.pos_rew_scale

        quat_rew = 1.0/(1 + quat_error**2)
        quat_rew *= self.quat_rew_scale

        u_rew = 1.0/(1 + u_error**2)
        u_rew *= self.u_rew_scale

        reward = pos_rew + quat_rew + u_rew

        return reward, pos_error, quat_error, u_error

    def get_est_reward(self, obs):
        if not len(obs.shape) == 2:
            raise RuntimeError('expecting size of 2 here')
        
        pos = obs[:, 0:3] * self.pos_std + self.pos_mean
        quat = obs[:, 3:7]
        u = self.unnormalize_action(obs[:, 13:17])
        reward, _, _, _ = self.compute_reward(pos, quat, u, single_axis=False)

        return reward



    # def get_est_reward(self, obs):
    #     #TODO put this in common function
    #     pos_error = np.linalg.norm(obs[:, 0:3] - self.target_traj[self.num_steps + 1, 0:3], axis=1)
    #     quat_error = np.linalg.norm(obs[:, 3:7] - self.target_traj[self.num_steps + 1, 3:7], axis=1)
        
    #     pos_rew = 1.0/(1 + pos_error**2)
    #     pos_rew *= self.pos_rew_scale

    #     quat_rew = 1.0/(1 + quat_error**2)
    #     quat_rew *= self.quat_rew_scale

    #     reward = pos_rew + quat_rew 

    #     return reward