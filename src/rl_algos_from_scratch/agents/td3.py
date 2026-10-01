import gymnasium as gym
import highway_env

gym.register_envs(highway_env)

def td3():

    env = gym.make("parking-v0",render_mode="human")
    obs , info = env.reset()
    truncated , terminated = False, False
    while not truncated and not terminated:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        print(obs["observation"])
        env.render()

