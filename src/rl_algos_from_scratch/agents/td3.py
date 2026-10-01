import gym


def main():
    env = gym.make("CarRacing-v3")
    info, _ = env.reset()
    print(info)
