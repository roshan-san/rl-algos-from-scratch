import gymnasium as gym
import highway_env

gym.register_envs(highway_env)


def make_env():
    env = gym.make("parking-v0", render_mode="human")
    return env
