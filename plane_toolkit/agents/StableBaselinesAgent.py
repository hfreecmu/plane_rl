import os
from plane_toolkit.agents.BaseAgent import BaseAgent
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import torch

class BaseStableBaselinesAgent(BaseAgent):
    def __init__(self, env):
        pass
    
    def train(self, env, cfg):
        model_dir = cfg["model_dir"]
        log_dir = cfg["log_dir"]
        num_par = cfg["num_par"]
        save_freq = cfg["save_freq"]
        total_timesteps = cfg["total_timesteps"]

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        dummy_env = DummyVecEnv([lambda: env] * num_par)
        self._create_model(cfg, dummy_env)

        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=log_dir,
            name_prefix=self.model_name + "_checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )

        self.model.learn(
            total_timesteps=int(total_timesteps),
            progress_bar=True,
            callback=checkpoint_callback,
        )

        model_path = os.path.join(model_dir, self.model_name)
        self.model.save(model_path)

    def load_model(self, model_dir):
        model_path = os.path.join(model_dir, self.model_name)
        self.model = self.rl_alg_class.load(model_path)

    def get_action(self, observation, env=None):
        if self.model is None:
            raise RuntimeError("Need to load model before getting action")

        return self.model.predict(observation, deterministic=True)
    
    def _create_model(self, cfg, env):
        """This needs to be defined in the subclass"""
        raise NotImplementedError()
    
class PPOAgent(BaseStableBaselinesAgent):
    def __init__(self, env):
        self.name = "PPO"
        self.policy = None
        self.model_name = "ppo_model"
        self.rl_alg_class = PPO
        self.model = None

    def _create_model(self, cfg, env):
        learning_rate = cfg["learning_rate"]
        n_steps = cfg["n_steps"]
        verbose = cfg["verbose"]

        # policy_kwargs = dict(activation_fn=torch.nn.ReLU,
        #                      net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])])

        policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                             net_arch=[dict(pi=[32, 32, 8], vf=[32, 32, 8])])

        #TODO should squash?

        self.model = self.rl_alg_class(
            self.policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=256,#64,#256,
            n_epochs = 8,#10, #new
            gamma = 0.99,
            gae_lambda = 0.95,
            clip_range = 0.2,
            clip_range_vf = 0.2,
            normalize_advantage=True,
            ent_coef=0.00,
            vf_coef=1.0, #0.5, #unsure what it is in rl_games, but skrl 1.0
            max_grad_norm=1.0, #0.5,
            use_sde=False,#True,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
        )

class TD3Agent(BaseStableBaselinesAgent):
    def __init__(self, env):
        self.name = "TD3"
        self.policy = None
        self.model_name = "td3_model"
        self.rl_alg_class = TD3
        self.model = None

    def _create_model(self, cfg, env):
        learning_rate = cfg["learning_rate"]
        verbose = cfg["verbose"]

        self.model = self.rl_alg_class(
            self.policy,
            env,
            learning_rate=1e-3,#learning_rate,
            buffer_size=1000000,#10000,
            learning_starts=10000, 
            batch_size=128, 
            #tau=0.01, 
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            verbose=verbose,
        )