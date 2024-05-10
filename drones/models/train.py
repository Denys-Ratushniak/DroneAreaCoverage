import torch
from torch import optim
from environment import DroneEnv
from rectangle import Rectangle
from actor_critic import Actor, Critic
import numpy as np

import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


def select_action(env, state, actor, epsilon, n_drones):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Adding batch dimension
        with torch.no_grad():  # Ensuring no gradients are computed during inference
            action_probs = actor(state_tensor)
            actions = torch.multinomial(action_probs.view(-1, 3), 1).view(-1, 1).numpy()  # Sample actions
        return actions.reshape(1, -1).squeeze()


def train(env, actor, critic, actor_optimizer, critic_optimizer, max_episode_time, delta_time, num_episodes):
    iterations = int(max_episode_time / delta_time)
    gamma = 0.99  # Discount factor for future rewards
    buffer = ReplayBuffer(10000)

    for episode in range(num_episodes):
        print(f"Training episode {episode + 1} of {num_episodes}")
        state = env.reset()
        total_reward = 0

        for i in range(iterations):
            # Convert the state to tensor and get action probabilities from the actor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
            action_probs = actor(state_tensor)

            # Sample actions from the probability distributions over actions
            action = torch.multinomial(action_probs.view(-1, 3), 1).view(-1, 1).numpy().squeeze()

            # TODO : denys 1) replay memory to store data for mini batches
            # TODO : denys 2) fix this function to be able to train on mini batches
            # Flatten to [batch * n_drones * 2, 3] and reshape back to [n_drones, 2]

            # Perform a step in the environment
            next_state, reward, done, _ = env.step(action)
            total_reward = reward

            # Convert next_state to tensor
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            # Compute the value (V(s)) and the value of the next state (V(s'))
            value = critic(state_tensor)
            next_value = critic(next_state_tensor)

            # Calculate the target and the advantage
            target_value = reward + gamma * next_value * (1 - int(done))  # Only consider future rewards if not done
            advantage = target_value - value
            # print(state_tensor)
            # print(next_state_tensor)
            # print(value)
            # print(next_value)
            # print("VALUE", target_value, value)


            # Calculate the critic loss using Mean Squared Error
            critic_loss = advantage.pow(2).mean()

            # Backpropagate the critic loss
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Calculate the actor loss
            log_prob = torch.log(action_probs.view(-1, 3) + 1e-10)  # Flatten and stabilize log

            chosen_log_probs = log_prob.gather(1, torch.tensor(action).reshape(-1, 1))
            print("before", chosen_log_probs.shape, chosen_log_probs)
            chosen_log_probs = chosen_log_probs.sum(dim=0)
            print("after", chosen_log_probs.shape, chosen_log_probs)
            actor_loss = - (chosen_log_probs * advantage.detach()).mean()  # Negative for gradient ascent

            print("l")
            print(action)
            print(action_probs)
            print(action_probs.view(-1, 3))
            print(log_prob)
            print(torch.tensor(action).reshape(-1, 1))
            print(chosen_log_probs)
            print(chosen_log_probs.shape)
            print(chosen_log_probs * advantage.detach())
            print(actor_loss)

            # Backpropagate the actor loss
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Update state
            state = next_state

            # Check if the episode has finished
            if done:
                break

        print(f"Episode {episode + 1} finished with total reward: {total_reward}")


if __name__ == '__main__':
    drones_cnt = 2
    dt = 0.07
    np.set_printoptions(precision=2)
    torch.set_printoptions(precision=2)
    save = True

    env = DroneEnv(
        n_drones=drones_cnt,
        delta_time=dt,
        render_mode=DroneEnv.performance,
        rectangle=Rectangle((0, 0), (1000, 1000), (50, 50))
    )

    env.reset()

    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]  # Adjust accordingly
    action_dim = env.action_space.shape[0]  # Adjust for your action space

    print(state_dim, action_dim)
    print(env.action_space.shape)

    actor = Actor(state_dim, drones_cnt)
    critic = Critic(state_dim)

    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    train(env, actor, critic, actor_optimizer, critic_optimizer, max_episode_time=20, delta_time=dt, num_episodes=100)

    if save:
        torch.save(actor.state_dict(), '../actor_model.pth')
        torch.save(critic.state_dict(), '../critic_model.pth')
