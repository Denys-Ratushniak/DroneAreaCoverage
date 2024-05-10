import torch
from torch import optim
from environment import DroneEnv
from rectangle import Rectangle
from simple_dqn import DQN
import numpy as np
from collections import namedtuple
from replay_memory import ReplayMemory
import random
import math

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()  # generate a random number between 0 and 1
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        # if the random number is greater than the threshold, select the action with the highest Q-value
        with torch.no_grad():
            return torch.argmax(policy_net(state.unsqueeze(0)), dim=-1).squeeze(0)
    else:  # else, select a random action from the environment's action space
        return env.sample_action()


def optimize_model():
    # If there are not enough transitions stored in the memory buffer to form a minibatch, return without doing anything
    if len(memory) < BATCH_SIZE:
        return

    state_batch, action_batch, next_state_batch, reward_batch = memory.sample(BATCH_SIZE)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action = policy_net(state_batch)

    state_action_values = policy_net(state_batch).gather(dim=3, index=action_batch.unsqueeze(-1)).squeeze(-1)
    # current Q

    # # Compute V(s_{t+1}) for all next states.
    # # Expected values of actions for non_final_next_states are computed based
    # # on the "older" target_net; selecting their best reward with max(1)[0].
    # # This is merged based on the mask, such that we'll have either the expected
    # # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values = target_net(next_state_batch).max(3)[0]

    # # Compute the expected Q values
    expected_state_action_values = reward_batch + (GAMMA * next_state_values)  # r + dis * Q(s')

    # # Compute the Huber loss between the current state-action values and the expected state-action values
    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


def train():
    num_episodes = 500
    average_rewards = []
    optimal_action_percentages = []

    for i_episode in range(num_episodes):
        state = env.reset()

        optimal_action_count = 0
        total_reward = 0

        for t in range(int(MAX_TIME_PER_EPISODE / DELTA_TIME)):
            action = select_action(state)

            with torch.no_grad():
                optimal_action = torch.argmax(policy_net(state.unsqueeze(0)), dim=-1).squeeze(0)

                print(state)
                print(action)
                print(optimal_action)
                print(torch.eq(action, optimal_action))

                identical_per_batch = (action == optimal_action).all(dim=0)

                optimal_action_count += identical_per_batch.sum().item()

                tensor_a = torch.zeros((2, 2, 2), dtype=torch.bool)
                tensor_b = torch.rand((2, 2, 2)) > 0.5

                print("\n\n\n", tensor_a)
                print(tensor_b)
                print((tensor_a == tensor_b).all(dim=2))
                print((tensor_a == tensor_b).all(dim=2).all(dim=1))
                print((tensor_a == tensor_b).all(dim=2).all(dim=1).sum()) # TODO:denys do i even need this? maybe just add 1 in select_action?

                # print((tensor_a == tensor_b).all(dim=0).sum().item())

            print(optimal_action_count)
            return



if __name__ == "__main__":
    BATCH_SIZE = 1
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4
    DRONES_CNT = 2
    DELTA_TIME = 0.07
    MAX_TIME_PER_EPISODE = 10

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print(f"using device: {device}")

    env = DroneEnv(
        n_drones=DRONES_CNT,
        delta_time=DELTA_TIME,
        render_mode=DroneEnv.human,
        rectangle=Rectangle((0, 0), (1000, 1000), (50, 50))
    )

    env.reset()

    # bad
    policy_net = DQN(DRONES_CNT, 6, 2).to(device)
    target_net = DQN(DRONES_CNT, 6, 2).to(device)

    # Load the weights of the policy network to the target network
    target_net.load_state_dict(policy_net.state_dict())

    # Define the replay memory transition as a named tuple
    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    # Initialize the optimizer
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR)

    memory = ReplayMemory(10000)

    state = env.state()
    action = select_action(state)
    # env.step(action)
    memory.push(state, action, env.step(action)[0], 5)
    memory.push(state, action, env.step(action)[0], 1)
    # print(memory.sample(1))
    # optimize_model()
    train()
    # print(state)
    # print(action)
    # print(env.state())
    # print(state_tensor)
