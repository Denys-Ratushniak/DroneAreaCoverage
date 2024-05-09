import time

import numpy as np
from winged_drone import *


class Rectangle:
    def __init__(self, lower_left, upper_right, grid_resolution):
        assert lower_left[0] < upper_right[0]
        assert lower_left[1] < upper_right[1]

        self.lower_left = np.array(lower_left)
        self.upper_right = np.array(upper_right)
        self.grid_resolution = np.array(grid_resolution)

    def center(self):
        return (self.lower_left + self.upper_right) / 2

    def width(self):
        return self.upper_right[0] - self.lower_left[0]

    def height(self):
        return self.upper_right[1] - self.lower_left[1]

    def area(self):
        return self.width() * self.height()

    def coverage_percentage(self, drones_list):
        covered_points_cnt = 0

        # Create a grid of points within the rectangle
        for x in np.linspace(self.lower_left[0], self.upper_right[0], self.grid_resolution[0]):
            for y in np.linspace(self.lower_left[1], self.upper_right[1], self.grid_resolution[1]):
                point = np.array([x, y])
                for drone in drones_list:
                    if np.linalg.norm(point - np.array([drone.x, drone.y])) <= drone.coverage_radius:
                        covered_points_cnt += 1
                        break

        # Calculate coverage
        total_points = self.grid_resolution[0] * self.grid_resolution[1]
        covered_count = covered_points_cnt
        return (covered_count / total_points) * 100

    def coverage_percentage_fast(self, drones_list):
        x = np.linspace(self.lower_left[0], self.upper_right[0], self.grid_resolution[0])
        y = np.linspace(self.lower_left[1], self.upper_right[1], self.grid_resolution[1])
        grid_x, grid_y = np.meshgrid(x, y)
        grid_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

        coverage_mask = np.zeros(grid_points.shape[0], dtype=bool)

        for drone in drones_list:
            drone_pos = np.array([drone.x, drone.y])
            distances = np.linalg.norm(grid_points - drone_pos, axis=1)
            coverage_mask |= (distances <= drone.coverage_radius)  # Update mask with OR operation

        covered_points_cnt = np.sum(coverage_mask)
        coverage_area = covered_points_cnt / coverage_mask.size * 100
        return coverage_area


if __name__ == "__main__":
    rectangle = Rectangle((0, 0), (10, 10), (100, 100))
    drones = [WingedDrone(5, 5, 0, 0, 0),
              WingedDrone(7, 7, 0, 0, 0),
              WingedDrone(2, 8, 0, 0, 0),
              WingedDrone(5, 8, 0, 0, 0),
              WingedDrone(2, 8, 0, 0, 0),
              WingedDrone(2, 8, 0, 0, 0),
              WingedDrone(3, 8, 0, 0, 0),
              WingedDrone(2, 8, 0, 0, 0)
              ]
    for drone in drones:
        drone.coverage_radius = 1  # Set the coverage radius for each drone

    print("Width of the rectangle:", rectangle.width())
    print("Height of the rectangle:", rectangle.height())
    print("Area of the rectangle:", rectangle.area())
    start_time = time.time()
    print("Coverage percentage:", rectangle.coverage_percentage(drones), "%")
    print(f"Default algorithm spent {time.time() - start_time:} seconds")
    start_time = time.time()
    print("Coverage percentage:", rectangle.coverage_percentage_fast(drones), "%")
    print(f"Fast algorithm spent {time.time() - start_time:} seconds")
