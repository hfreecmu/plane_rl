from gym.envs.registration import register

register(
    id='plane-v0',
    entry_point='gym_plane.envs:PlaneEnv',
)