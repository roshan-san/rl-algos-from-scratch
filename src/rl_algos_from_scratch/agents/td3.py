import gymnasium as gym
import highway_env

from rl_algos_from_scratch.networks.Td3Actor import Td3Actor
from rl_algos_from_scratch.networks.Td3Critic import Td3Critic

gym.register_envs(highway_env)


def td3():

    env = gym.make("parking-v0", render_mode="human")
    obs, info = env.reset()
    truncated, terminated = False, False

    input_dim = 4
    output_dim =2

    actor_net = Td3Actor(input_dim, (400,300),output_dim)

    critic_net  = Td3Critic(input_dim +output_dim , (400,300),output_dim)



    while not truncated and not terminated:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        print(obs["observation"])
        env.render()
