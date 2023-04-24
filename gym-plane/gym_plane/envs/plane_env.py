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

        #terminate position
        self.pos_tol = info_dict["pos_tol"]

        self.pos_rew_scale = 1.0#info_dict["pos_rew_scale"]
        self.quat_rew_scale = 1.0#info_dict["quat_rew_scale"]
        self.u_rew_scale = 0.0#info_dict["u_rew_scale"]
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
        self.clip_actions = 1.0

        self.clip_positions = 10.0
        self.clip_v = 10.0
        self.clip_w = 2*np.pi
        self.x_bound = self.obs_bound_factor*np.max(np.abs(self.ilqr_res))

        #for starters observation space will
        #13 for x
        #4 for last control vector
        #1 for timestep

        # self.observation_shape = (18,)
        # self.observation_space = gym.spaces.Box(
        #     low=-self.obs_bound, high=self.obs_bound, shape=self.observation_shape, dtype=np.float32,
        # )
        # self.obs = np.zeros(self.observation_shape, dtype=np.float32)
        
        #13 for state, 13 for next target
        self.observation_shape = (26,)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float32,
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

        #target position for this branch
        #self.target_pos = self.ilqr_res[51, 0:3]
        self.targets = self.ilqr_ref[1:, :]
        self.init_target = self.ilqr_ref[0, :]


    def reset(self):
        #initial state

        #self.x = np.copy(self.ilqr_res[0, :])
        self.x = np.copy(self.init_target)
        self.num_steps = 0

        self._make_observation() 

        #self.init_distance = np.linalg.norm(self.x[0:3] - self.target_pos)

        return np.copy(self.obs)

    def step(self, action):
        if not (len(action.shape) == 1):
            raise RuntimeError("dim issue again")
        
        action = np.clip(action, -self.clip_actions, self.clip_actions)

        #rescale action to get us
        last_u = self.unnormalize_action(action)

        #update x
        x_new = normalized_rk4_step(self.yak, self.x, last_u, self.h)

        if not len(x_new.shape) == 1:
            raise RuntimeError("dim issue again")
        
        if np.isnan(x_new).sum():
            raise RuntimeError('x_new isnan')
        
        #unstable flight
        if (np.max(np.abs(x_new)) >= self.x_bound):
            reward, pos_error, quat_error, _ = self.compute_reward(x_new)
            #reward = -100
            
            self.x = x_new
            self.num_steps += 1
            done = True

            self._make_observation()

            obs = np.copy(self.obs)
            info = {}

            self.latest_pos_error = pos_error
            self.latest_quat_error = quat_error
            self.latest_u_error = 0

            return obs, reward, done, info

        reward, pos_error, quat_error, reset = self.compute_reward(x_new)

        self.x = x_new
        self.num_steps += 1
        if reset:
            done = True
        else:
            done = self.num_steps >= self.N

        self._make_observation()

        obs = np.copy(self.obs)
        info = {}

        self.latest_pos_error = pos_error
        self.latest_quat_error = quat_error
        self.latest_u_error = 0

        return obs, reward, done, info
  
    def render(self):
        pass

    def _make_observation(self):
        self.obs[0:3] = np.clip(self.x[0:3], -self.clip_positions, self.clip_positions)
        self.obs[3:7] = self.x[3:7]
        self.obs[7:10] = np.clip(self.x[7:10], -self.clip_v, self.clip_v)
        self.obs[10:13] = np.clip(self.x[10:13], -self.clip_w, self.clip_w)
        
        self.obs[0:3] = (self.obs[0:3] - self.pos_mean) / self.pos_std
        self.obs[3:7] = self.obs[3:7] / np.linalg.norm(self.obs[3:7])
        self.obs[7:10] = (self.obs[7:10] - self.v_mean) / self.v_std
        self.obs[10:13] = (self.obs[10:13] - self.w_mean) / self.w_std

        if self.num_steps < self.N:
            target = self.targets[self.num_steps, :]
        else:
            target = self.targets[self.num_steps - 1, :]

        self.obs[13:16] = np.clip(target[0:3], -self.clip_positions, self.clip_positions)
        self.obs[16:20] = target[3:7]
        self.obs[20:23] = np.clip(target[7:10], -self.clip_v, self.clip_v)
        self.obs[23:26] = np.clip(target[10:13], -self.clip_w, self.clip_w)
        
        self.obs[13:16] = (self.obs[13:16] - self.pos_mean) / self.pos_std
        self.obs[16:20] = self.obs[16:20] / np.linalg.norm(self.obs[16:20])
        self.obs[20:23] = (self.obs[20:23] - self.v_mean) / self.v_std
        self.obs[23:26] = (self.obs[23:26] - self.w_mean) / self.w_std  
            
    def normalize_action(self, u):
        return 2*(u - self.action_min) / (self.action_max - self.action_min) - 1
    
    def unnormalize_action(self, action):
        u = ((action + 1) / 2) * (self.action_max - self.action_min) + self.action_min
        return u
    
    def compute_reward(self, x, single_axis=True):
        pos = x[0:3]
        quat = x[3:7]
        lin_vel = x[7:10]
        ang_vel = x[10:13]
        if single_axis:
            #pos_error = np.linalg.norm(pos - self.ilqr_res[self.num_steps + 1, 0:3])
            pos_error = np.linalg.norm(pos - self.targets[self.num_steps, 0:3])
            quat_error = calc_quat_error(quat, self.targets[self.num_steps, 3:7])
            lin_vel_error = np.linalg.norm(lin_vel - self.targets[self.num_steps, 7:10])
            ang_vel_error = np.linalg.norm(ang_vel - self.targets[self.num_steps, 10:13])
        else:
            raise RuntimeError('not supported with canges')
            pos_error = np.linalg.norm(pos - self.ilqr_res[self.num_steps + 1, 0:3], axis=1)
            
            quat_error = np.zeros(pos_error.shape)
            for i in range(quat_error.shape[0]):
                quat_error[i] = calc_quat_error(quat[i], self.ilqr_res[self.num_steps + 1, 3:7])
            u_error = np.linalg.norm(u - self.ilqr_control[self.num_steps], axis=1)

        pos_rew = 1.0/(1.0 + pos_error**2)
        pos_rew *= self.pos_rew_scale

        quat_rew = 1.0/(1.0 + quat_error**2)
        quat_rew *= self.quat_rew_scale

        lin_vel_rew = 1.0/(1.0 + lin_vel_error**2)
        ang_vel_rew = 1.0/(1.0 + ang_vel_error**2)

        #quat, lin_vel, and ang_vel rewards only matter when close
        reward = pos_rew + pos_rew * (quat_rew + lin_vel_rew + ang_vel_rew)

        #reset when error is too far
        reset = (pos_error > self.pos_tol)

        return reward, pos_error, quat_error, reset
