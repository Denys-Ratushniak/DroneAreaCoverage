import torch
from torch import optim
from environment import DroneEnv
from rectangle import Rectangle
from actor_critic import Actor, Critic
import numpy as np


def train(env, actor, critic, actor_optimizer, critic_optimizer, max_episode_time, delta_time, num_episodes):
    iterations = int(max_episode_time / delta_time)

    for episode in range(num_episodes):
        print(f"Training episode {episode+1} of {num_episodes}")
        state = env.reset()
        done = False
        for i in range(iterations):
            state_tensor = torch.FloatTensor(state)
            action_probs = actor(state_tensor)

            action = torch.multinomial(action_probs, 1).item()

            next_state, reward, done, _ = env.step(action)

            # Update Critic
            value = critic(state_tensor)
            next_state_tensor = torch.FloatTensor(next_state)
            next_value = critic(next_state_tensor)
            target_value = reward + 0.99 * next_value
            critic_loss = (value - target_value.detach()).pow(2).mean()  # Ensure scalar

            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # Update Actor
            advantage = (target_value - value).detach()
            actor_loss = -torch.log(action_probs[action]) * advantage
            actor_loss = actor_loss.mean()  # Ensure scalar

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            state = next_state

            if i == 0 or i == iterations - 1:
                print(f"Iteration: {i+1}, State: {state}")

            if done or i == iterations - 1:
                print(f"End of episode or iteration reached: {i+1}")
                break


if __name__ == '__main__':
    drones_cnt = 1
    dt = 0.07
    np.set_printoptions(precision=2)

    env = DroneEnv(
        n_drones=drones_cnt,
        delta_time=dt,
        render_mode=DroneEnv.performance,
        rectangle=Rectangle((0, 0), (1000, 1000), (50, 50))
    )

    env.reset()

    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1] # Adjust accordingly
    action_dim = env.action_space.shape[0]  # Adjust for your action space

    print(state_dim, action_dim)
    print(env.action_space.shape)

    actor = Actor(state_dim, drones_cnt)
    critic = Critic(state_dim)

    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-3)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)

    train(env, actor, critic, actor_optimizer, critic_optimizer, max_episode_time=5, delta_time=dt, num_episodes=10)
