from plane_toolkit.agents.StableBaselinesAgent import (
    PPOAgent,
    TD3Agent,
)
from plane_toolkit.agents.MBAgent import MBAgent

agent_dict = {
    "PPO": PPOAgent,
    "TD3": TD3Agent,
    "MB": MBAgent,
}
