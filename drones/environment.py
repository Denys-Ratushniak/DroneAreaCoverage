import gym
import numpy as np
import matplotlib.pyplot as plt
import time
from gym.spaces import MultiDiscrete
import torch
from winged_drone import WingedDrone
from rectangle import Rectangle
from models import *


class DroneEnv(gym.Env):
    human = "human_mode"
    performance = "performance_mode"
    __render_modes = [human, performance]

    def __init__(self, n_drones, delta_time, render_mode, rectangle):
        if render_mode not in self.__render_modes:
            print('Invalid render mode')
            exit(0)

        super(DroneEnv, self).__init__()

        self.n_drones = n_drones
        self.delta_time = delta_time
        self.render_mode = render_mode
        self.area_rectangle = rectangle

        self.drones = [WingedDrone() for _ in range(n_drones)]

        self.action_space = MultiDiscrete([3, 3] * n_drones)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(n_drones, 6), dtype=np.float32)

        self.colors = ["green", "yellow", "purple", "blue", "red"][:n_drones]

        self.scatter = None
        self.figure, self.ax = plt.subplots(figsize=(5, 5))
        self.plot_drones()
        plt.ion()
        plt.show()

    def drone_state_numbers_cnt(self):
        return len(np.array(self.drones[0].get_state()))

    def state(self):
        return np.array([drone.get_state() for drone in self.drones])

    def step(self, actions):
        for i in range(self.n_drones):
            speed_change = actions[2 * i] - 1  # Map {0, 1, 2} to {-1, 0, 1}
            omega_change = (actions[2 * i + 1] - 1) / 10  # Map {0, 1, 2} to {-0.1, 0, 0.1}

            self.drones[i].update_state(speed_change, omega_change, self.delta_time)

        reward = 0

        for drone in self.drones:
            center = self.area_rectangle.center()
            drone_pos = np.array([drone.x, drone.y])
            distance = np.linalg.norm(drone_pos - center)
            reward += -distance

        return self.state(), reward, False, {}

    def update_action(self):
        action = self.action_space.sample()
        self.step(action)

    def plot_drones(self):
        self.scatter = self.ax.scatter(
            [drone.x for drone in self.drones],
            [drone.y for drone in self.drones],
            c=self.colors,
            s=100
        )
        self.plot_drones_area()

    def plot_drones_area(self):
        for i, drone in enumerate(self.drones):
            circle = plt.Circle(
                (drone.x, drone.y),
                WingedDrone.coverage_radius,
                color=self.colors[i],
                fill=True,
                alpha=0.2
            )
            self.ax.add_patch(circle)

    def plot_rectangle(self):
        rect = plt.Rectangle(
            self.area_rectangle.lower_left,
            self.area_rectangle.width(),
            self.area_rectangle.height(),
            fill=False,
            edgecolor='red'
        )

        self.ax.add_patch(rect)

    def plot_coverage(self):
        coverage_percent = self.area_rectangle.coverage_percentage_fast(self.drones)
        self.ax.text(
            0.5,
            0.02,
            f'Coverage: {coverage_percent:.2f}%',
            transform=self.ax.transAxes,
            color='black',
            fontsize=12
        )

    def render(self):
        if self.render_mode == self.human:
            if not plt.fignum_exists(self.figure.number):
                return False
            self.ax.clear()

            self.plot_drones()
            self.plot_rectangle()
            self.plot_coverage()

            self.ax.set_xlim(-1000, 2000)
            self.ax.set_ylim(-1000, 2000)

            self.figure.canvas.draw()
            self.figure.canvas.flush_events()

        return True

    def reset(self, **kwargs):
        for drone in self.drones:
            drone.reset_random()

        return self.state()

    def run(self):
        target_time = self.delta_time
        while True:
            start_time = time.time()
            self.update_action()
            if not self.render():
                break

            elapsed_time = time.time() - start_time
            if elapsed_time > target_time:
                print("latency", elapsed_time - target_time)
            sleep_time = max(0.0, target_time - elapsed_time)
            time.sleep(sleep_time)

    def reset_and_run_with_models(self, actor_path, critic_path):
        input_dim = self.observation_space.shape[0] * self.observation_space.shape[1]

        actor = Actor(input_dim=input_dim, n_drones=self.n_drones)
        critic = Critic(input_dim=input_dim)

        actor.load_state_dict(torch.load(actor_path))
        critic.load_state_dict(torch.load(critic_path))

        actor.eval()
        critic.eval()

        target_time = self.delta_time
        state = self.reset()
        while True:
            start_time = time.time()

            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Adding batch dimension
            with torch.no_grad():  # Ensuring no gradients are computed during inference
                action_probs = actor(state_tensor)
                actions = torch.multinomial(action_probs.view(-1, 3), 1).view(-1, 1).numpy()  # Sample actions

            action_for_env = actions.reshape(1, -1).squeeze()

            next_state, reward, done, _ = self.step(action_for_env)
            state = next_state

            if not self.render():
                break

            elapsed_time = time.time() - start_time
            if elapsed_time > target_time:
                print("latency", elapsed_time - target_time)
            sleep_time = max(0.0, target_time - elapsed_time)
            time.sleep(sleep_time)


if __name__ == "__main__":
    env = DroneEnv(
        n_drones=3,
        delta_time=0.07,
        render_mode=DroneEnv.human,
        rectangle=Rectangle((0, 0), (1000, 1000), (50, 50))
    )

    env.reset()
    env.run()
