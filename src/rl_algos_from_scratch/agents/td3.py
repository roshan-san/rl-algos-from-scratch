import torch

from rl_algos_from_scratch.common.env import make_env
from rl_algos_from_scratch.memory.BufferReplay import BufferReplay
from rl_algos_from_scratch.networks.Td3Actor import Td3Actor
from rl_algos_from_scratch.networks.Td3Critic import Td3Critic


class Td3Agent:
    def __init__(self, state_dim, action_dim, gamma):
        self.actor_net = Td3Actor(state_dim, (400, 300), action_dim)

        self.critic_net_1 = Td3Critic(state_dim + action_dim, (400, 300), 1)
        self.critic_net_2 = Td3Critic(state_dim + action_dim, (400, 300), 1)

        self.target_actor_net = Td3Actor(state_dim, (400, 300), action_dim)
        self.target_critic_net_1 = Td3Critic(state_dim + action_dim, (400, 300), 1)
        self.target_critic_net_2 = Td3Critic(state_dim + action_dim, (400, 300), 1)

        self.target_critic_net_1.load_state_dict(self.critic_net_1.state_dict())
        self.target_critic_net_2.load_state_dict(self.critic_net_2.state_dict())
        self.target_actor_net.load_state_dict(self.actor_net.state_dict())

        self.actor_optim = torch.optim.Adam
        self.critic_optim = torch.optim.Adam

        self.critic_loss_fn = torch.nn.MSELoss()

        self.gamma = gamma

    def act(self, state, explore):

        action = self.actor_net(state)

        if not explore:
            pass

        return action

    def update(self, batch):

        state, action, new_state, reward, done = batch

        # actor loss

        action_online = self.actor_net(state)
        q_val = min(
            self.critic_net_1(state, action_online),
            self.critic_net_2(state, action_online),
        )
        actor_loss = -q_val.mean()

        next_action = self.target_actor_net(new_state)
        td_target1 = reward + (1 - done) * self.gamma * self.target_critic_net_1(
            new_state, next_action
        )
        td_target2 = reward + (1 - done) * self.gamma * self.target_critic_net_2(
            new_state, next_action
        )

        c1_loss = self.critic_loss_fn(self.critic_net_1(state, action), td_target1)
        c2_loss = self.critic_loss_fn(self.critic_net_2(state, action), td_target2)

        critic_loss = c1_loss + c2_loss

    def save(self):
        pass

    def load(self):
        pass


def train():
    epoch = 1000
    mem_size = 10000
    sample_size = 250
    gamma = 0.99

    env = make_env()
    action_dim = 2
    state_dim = 4

    memory = BufferReplay(mem_size, state_dim, action_dim)
    agent = Td3Agent(state_dim, action_dim, gamma)

    for e in range(epoch):
        state, _ = env.reset()
        truncated, terminated = False, False

        while not truncated and not terminated:
            action = agent.act(state, explore=True)
            new_state, reward, terminated, truncated, _ = env.step(action)
            memory.store(state, action, reward, new_state, truncated or terminated)

            if len(memory) > mem_size:
                batch = memory.sample(sample_size)
                agent.update(batch)
            state = new_state
