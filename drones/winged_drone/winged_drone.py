import numpy as np
import math


class WingedDrone:
    acceleration = 0.1
    min_speed = 0.5
    max_speed = 5.0
    max_omega = 0.5
    position_bounds = [0, 10]
    coverage_radius = 1

    def __init__(self, x=0, y=0, vx=0, vy=0, omega=0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.omega = omega
        self.theta = 0

    def move(self, dt):
        self.theta += self.omega * dt
        self.theta = self.theta % (2 * math.pi)  # it works

        speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
        self.vx = math.cos(self.theta) * speed
        self.vy = math.sin(self.theta) * speed

        self.x += self.vx * dt
        self.y += self.vy * dt
        self.x = np.clip(self.x, 0, 10)
        self.y = np.clip(self.y, 0, 10)

    def change_speed(self, d_speed):
        speed_change = self.acceleration * d_speed
        speed = math.sqrt(self.vx ** 2 + self.vy ** 2) + speed_change
        speed = np.clip(speed, self.min_speed, self.max_speed)

        # TODO would be nice to not clip min but penalize it if drone is in state covering
        # TODO also maybe penalize for action that has no sense? v = max_speed and increase max_speed

        self.vx = math.cos(self.theta) * speed
        self.vy = math.sin(self.theta) * speed

    def change_omega(self, d_omega):
        self.omega = np.clip(self.omega + d_omega, -self.max_omega, self.max_omega)

    def update_state(self, d_speed_factor, d_omega, dt):
        self.change_speed(d_speed_factor)
        self.change_omega(d_omega)
        self.move(dt)

    def get_state(self):
        return np.array([self.x, self.y, self.vx, self.vy, self.theta, self.omega])

    def reset_zero(self):
        self.x, self.y, self.vx, self.vy, self.omega, self.theta = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    def reset_random(self):
        self.x = np.random.uniform(*self.position_bounds)
        self.y = np.random.uniform(*self.position_bounds)

        speed = np.random.uniform(self.min_speed, self.max_speed)
        self.theta = np.random.uniform(0, 2 * math.pi)

        self.vx = math.cos(self.theta) * speed
        self.vy = math.sin(self.theta) * speed

        self.omega = np.random.uniform(-self.max_omega, self.max_omega)
