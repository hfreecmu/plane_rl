import os
from plane_toolkit.agents.BaseAgent import BaseAgent
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

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

        self.model = self.rl_alg_class(
            self.policy,
            env,
            n_steps=n_steps,
            learning_rate=learning_rate,
            verbose=verbose,
            ent_coef=0.00,
            use_sde=True,
            max_grad_norm=0.5,
            n_epochs=20,
            batch_size=256,
        )